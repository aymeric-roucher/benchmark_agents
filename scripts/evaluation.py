import os
import re
import json
from langchain.llms import HuggingFaceEndpoint
from langchain.prompts.chat import ChatPromptTemplate
import pandas as pd
import asyncio
from typing import Optional, List
import tqdm.asyncio
import numpy as np
from threading import Thread
from queue import Queue
import datasets

_SENTINEL_KILL_CONSUMERS = object()


def build_evaluator(hf_endpoint_url: str) -> tuple:
    """
    Build an evaluator language model using the given Hugging Face endpoint URL.

    Args:
        hf_endpoint_url (str): The URL of the Hugging Face endpoint.

    Returns:
        Tuple: A tuple containing the evaluator chat model and the correctness prompt template.
    """
    eval_chat_model = HuggingFaceEndpoint(
        endpoint_url=hf_endpoint_url,
        task="text-generation",
        model_kwargs={
            "max_new_tokens": 488,
            "do_sample": False,
            "repetition_penalty": 1.03,
        },
    )
    return eval_chat_model


async def evaluate_single_example(
    example: dict, evaluator, eval_prompt_template, evaluator_name, eval_split_string="[RESULT]", writer_queue: Optional[Queue] = None
):
    if f"eval_score_{evaluator_name}" in example:
        try:
            el = float(example[f"eval_score_{evaluator_name}"])
            assert not np.isnan(el)
            return example
        except:
            pass
    eval_prompt = eval_prompt_template.format_messages(
        instruction=example["question"],
        response=example["prediction"],
        reference_answer=example["gt_answer"],
    )
    print("Evaluating example")
    eval_result = await evaluator.ainvoke(eval_prompt)
    eval_result = eval_result.content
    try:
        feedback, score = [item.strip() for item in eval_result.split(eval_split_string)]
    except:
        print(eval_result)
        segments = [
            segment.strip() for segment in eval_result.split(eval_split_string) if segment.strip()
        ]
        # Search for a segment that contains a numerical score
        for segment in segments:
            if segment.isdigit():
                feedback = ""
                score = int(segment)
    example[f"eval_score_{evaluator_name}"] = score
    example[f"eval_feedback_{evaluator_name}"] = feedback
    if writer_queue:
        writer_queue.put(example)
    return example


async def evaluate_answers(
    examples: List,
    evaluator,
    evaluator_name: str,
    eval_prompt_template: ChatPromptTemplate,
    eval_split_string: str = "[RESULT]",
    output_file_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run a full evaluation on the given dataset using multiple agent models.
    Uses safe writing in multithreading, from options suggested here:
    https://stackoverflow.com/questions/33107019/multiple-threads-writing-to-the-same-csv-in-python

    Args:
        dataset (Dataset): The dataset to test on.
        agents (Dict[str, AgentExecutor]): A dictionary of agent executors to test on the dataset

    Returns:
        pd.DataFrame: The evaluation results as a pandas DataFrame.
    """
    if output_file_path and os.path.isfile(output_file_path):
        previous_evaluations = pd.read_json(output_file_path, lines=True)
        if f"eval_score_{evaluator_name}" in previous_evaluations.columns:
            previous_evaluations = previous_evaluations.loc[previous_evaluations[f"eval_score_{evaluator_name}"].notna()]
            print('Previous evaluations:')
            
            examples = [example for example in examples if not len(previous_evaluations.loc[
                (previous_evaluations["question"] == example["question"]) & (previous_evaluations["agent_name"] == example["agent_name"])
            ]) > 0]

    print(f"Launching evaluation for {len(examples)} examples...")

    writer_queue = Queue()

    with open(output_file_path, "a") as output_file:
        def write_line():
            while True:
                if not writer_queue.empty():
                    annotated_example = writer_queue.get()
                    
                    if annotated_example is _SENTINEL_KILL_CONSUMERS:
                        writer_queue.put(_SENTINEL_KILL_CONSUMERS) # put it back so that other consumers see it
                        return
                    
                    annotated_example = {k: str(v) for k, v in annotated_example.items()}

                    # Row comes out of writer_queue; JSON writing goes here
                    json.dump(annotated_example, output_file)
                    output_file.write('\n')
        
        consumer = Thread(target=write_line)
        consumer.setDaemon(True)
        consumer.start()

        tasks = [
            evaluate_single_example(
                example,
                evaluator,
                eval_prompt_template,
                evaluator_name,
                eval_split_string,
                writer_queue,
            )
            for example in examples
        ]

        evaluation_results = [await f for f in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks))]
        writer_queue.put(_SENTINEL_KILL_CONSUMERS)

    return evaluation_results


def extract_number(string):
    try:
        found_strings = [el.strip() for el in re.findall(r"(?:[,\d]+.?\d*)", string)]

        found_strings = [
            "".join(ch for ch in el if (ch.isalnum() or ch == "."))
            for el in found_strings
            if el[0].isdigit() or el[0] == "."
        ]
        found_strings = [el for el in found_strings if len(el) > 0]

        found_string = found_strings[-1]
        return float(found_string)
    except Exception as e:
        print("Error when extracting string:", e)
        return 0


def split_answer(row):
    splitted = row["answer"].split("####")
    row["true_reasoning"] = splitted[0]
    row["true_answer"] = float(splitted[1].strip())
    return row


def load_math_datasets():
    math_dataset = (
        datasets.load_dataset("gsm8k", "main")["train"].shuffle(seed=42).select(range(100))
    )
    math_dataset = pd.DataFrame(math_dataset)

    math_dataset = math_dataset.apply(split_answer, axis=1)
    math_dataset = math_dataset.drop(columns=["answer"])
    math_dataset = datasets.Dataset.from_pandas(math_dataset)

    eval_dataset = math_dataset.select(range(30))
    fewshot_dataset = math_dataset.select(range(10))
    return eval_dataset, fewshot_dataset