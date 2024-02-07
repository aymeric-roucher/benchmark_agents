import json
from langchain.llms import HuggingFaceEndpoint
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.chat_models import ChatOpenAI
import pandas as pd
import asyncio
from scripts.run_agents import run_agent
from typing import Dict, Optional, Any, Tuple, List
from langchain.agents import AgentExecutor
import tqdm.asyncio
import numpy as np


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
    example: dict, evaluator, eval_prompt_template, evaluator_name, eval_split_string="[RESULT]"
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
    return example


async def evaluate_answers(
    examples: List,
    evaluator,
    evaluator_name: str,
    eval_prompt_template: ChatPromptTemplate,
    eval_split_string: str = "[RESULT]",
    output_file: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run a full evaluation on the given dataset using multiple agent models.

    Args:
        dataset (Dataset): The dataset to test on.
        agents (Dict[str, AgentExecutor]): A dictionary of agent executors to test on the dataset

    Returns:
        pd.DataFrame: The evaluation results as a pandas DataFrame.
    """

    tasks = [
        evaluate_single_example(
            example,
            evaluator,
            eval_prompt_template,
            evaluator_name,
            eval_split_string,
        )
        for example in examples
    ]

    evaluation_results = [await f for f in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks))]
    if output_file:
        with open(output_file, "w") as f:
            json.dump(evaluation_results, f)

    return evaluation_results
