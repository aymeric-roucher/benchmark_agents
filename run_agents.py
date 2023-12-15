import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import json
import pandas as pd
from datasets import Dataset
from chat_wrapper import HuggingFaceChatWrapper, BaseChatModel
from langchain.agents import AgentExecutor, load_tools
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import (
    ReActJsonSingleInputOutputParser,
)
from langchain.llms import (
    HuggingFaceEndpoint,
)
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    SystemMessage,
)
from prompts import HUMAN_PROMPT, EVALUATION_PROMPT, SYSTEM_PROMPT
from tqdm import tqdm
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.tools.render import render_text_description_and_args, format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import pandas as pd


def build_openai_agent(tools, model_id: Optional[str] = 'gpt-4-1106-preview'):
    llm = ChatOpenAI(model=model_id, temperature=0)
    llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
    )
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )


def build_hf_agent(hf_endpoint_url: str, tools):
    """
    Build a zero-shot ReAct chat agent from HF endpoint.

    Args:
        hf_endpoint_url (str): The endpoint URL for the Hugging Face model.

    Returns:
        AgentExecutor: An agent executor object that can be used to run the agent.

    """
    # instantiate LLM and chat model
    llm = HuggingFaceEndpoint(
        endpoint_url=hf_endpoint_url,
        task="text-generation",
        model_kwargs={
            "max_new_tokens": 512,
            "do_sample": False,
            "repetition_penalty": 1.03,
        },
    )

    chat_model = HuggingFaceChatWrapper(llm=llm)

    prompt = ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template(SYSTEM_PROMPT+'\nSo, here is my question:'+HUMAN_PROMPT),
        ]
    )
    prompt = prompt.partial(
        tool_description=render_text_description_and_args(tools),
        tool_names=", ".join([t.name for t in tools]),
    )
    

    # define the agent
    chat_model_with_stop = chat_model.bind(stop=["\nObservation"])
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        }
        | prompt
        | chat_model_with_stop
        | ReActJsonSingleInputOutputParser()
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )


async def run_agent(
    question: str,
    ground_truth_answer: str,
    agent_executor: AgentExecutor,
    agent_name: str,
) -> dict:
    """
    Runs the execution and evaluation process for a given question and ground truth answer.

    Args:
        question (str): The input question to be evaluated.
        ground_truth_answer (str): The ground truth answer for the question.
        agent_executor (AgentExecutor): The agent executor object used to run the agent.
        evaluator (HuggingFaceChatWrapper): The evaluator object used to evaluate the agent's response.
        eval_prompt_template (ChatPromptTemplate): The template for the evaluation prompt.

    Returns:
        dict: A dictionary containing the evaluation results, including the agent model ID, evaluator model ID,
        question, ground truth answer, prediction, intermediate steps, evaluation score, evaluation feedback,
        tool call parsing error flag, iteration limit exceeded flag, and agent error (if any).
    """
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        # run executor agent
        out = await agent_executor.ainvoke({"input": question})

        # check for parsing errors which indicate the LLM failed to follow the ReACT format
        # this could be due to an issue with the tool calling format or ReACT formatting (i.e. Thought, Action, Observation, etc.)
        parsing_error = (
            True
            if any(
                [
                    "Could not parse LLM output" in step[0].log
                    for step in out["intermediate_steps"]
                ]
            )
            else False
        )

        # check if iteration limit exceeded
        iteration_limit_exceeded = (
            True
            if "Agent stopped due to iteration limit or time limit." in out["output"]
            else False
        )
        raised_exception = False

    except Exception as e:
        print('Error on ', agent_executor, question, e)
        out = {"output": None, "intermediate_steps": None}
        score = None
        feedback = None
        parsing_error = False
        iteration_limit_exceeded = False
        exception = e
        raised_exception = True

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # collect results
    return {
        "agent_name": agent_name,
        "agent_model_id": agent_executor.dict()["agent"]["runnable"]["middle"][-1][
            "bound"
        ]["_type"],
        "question": question,
        "gt_answer": ground_truth_answer,
        "prediction": out["output"],
        "intermediate_steps": out["intermediate_steps"],
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": repr(exception) if raised_exception else None,
        "start_time": start_time,
        "end_time": end_time,
    }


async def answer_questions(
    dataset: Dataset,
    agent_executor: AgentExecutor,
    agent_name: str,
) -> List[Dict[str, Any]]:
    """
    Evaluates the agent on a given dataset.

    Args:
        dataset (Dataset): The dataset to test the agent on.
        agent_executor (AgentExecutor): The agent executor object used to run the agent.
        agent_name (str): The name of the agent model.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the evaluation results for each example in the dataset.
        Each dictionary includes the agent model ID, evaluator model ID, question, ground truth answer, prediction,
        intermediate steps, evaluation score, evaluation feedback, tool call parsing error flag, iteration limit
        exceeded flag, agent error (if any), and example metadata (task).
    """
    try:
        with open(f'output/{agent_name}.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        results = []
        
    results_df = pd.DataFrame(results)

    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        if len(results_df) > 0:
            if example["question"] in results_df['question'].unique():
                continue

        # run agent
        result = await run_agent(
            question=example["question"],
            ground_truth_answer=example["answer"],
            agent_executor=agent_executor,
            agent_name=agent_name,
        )
        print("Result:", result)
        print("True answer:", example["answer"])

        # add in example metadata
        result.update(
            {
                "task": example["task"],
            }
        )
        results.append(result)

        with open(f'output/{agent_name}.json', 'w') as f:
            json.dump(results, f)
    return results


async def run_full_tests(
    dataset: Dataset,
    agents: Dict[str, AgentExecutor],
) -> pd.DataFrame:
    """
    Run a full evaluation on the given dataset using multiple agent models.

    Args:
        dataset (Dataset): The dataset to test on.
        agents (Dict[str, AgentExecutor]): A dictionary of agent executors to test on the dataset

    Returns:
        pd.DataFrame: The evaluation results as a pandas DataFrame.
    """
    results = []

    tasks = [
        answer_questions(
            dataset=dataset,
            agent_executor=agent_executor,
            agent_name=agent_name,
        )
        for agent_name, agent_executor in agents.items()
    ]

    results = await asyncio.gather(*tasks)

    return pd.DataFrame([element for sublist in results for element in sublist])
