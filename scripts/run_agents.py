import asyncio
from datetime import datetime
from typing import Any, Dict, List, Callable
import json
import pandas as pd
from tqdm import tqdm
from datasets import Dataset

from langchain.agents import AgentExecutor
from langchain.tools.base import ToolException


def acall_langchain_agent(agent: AgentExecutor, question: str) -> str:
    return agent.ainvoke({"input": question})

def call_langchain_agent(agent: AgentExecutor, question: str) -> str:
    return agent.invoke({"input": question})

async def arun_agent(
    question: str,
    agent_executor: AgentExecutor,
    agent_name: str,
    agent_call_function: Callable,
) -> dict:
    """
    Runs the execution process for a given question and ground truth answer.

    Args:
        question (str): The input question to be evaluated.
        agent_executor (AgentExecutor): The agent executor object used to run the agent.
        agent_name (str): The name of the agent model.

    Returns:
        dict: A dictionary containing the evaluation results, including the agent model ID, evaluator model ID,
        question, ground truth answer, prediction, intermediate steps, evaluation score, evaluation feedback,
        tool call parsing error flag, iteration limit exceeded flag, and agent error (if any).
    """
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        # run executor agent
        response = await agent_call_function(agent_executor, question)

        # check for parsing errors which indicate the LLM failed to follow the ReACT format
        # this could be due to an issue with the tool calling format or ReACT formatting (i.e. Thought, Action, Observation, etc.)
        parsing_error = (
            True
            if any(
                [
                    "Could not parse LLM output" in step[0].log
                    for step in response["intermediate_steps"]
                ]
            )
            else False
        )

        # check if iteration limit exceeded
        iteration_limit_exceeded = (
            True
            if "Agent stopped due to iteration limit or time limit." in response["output"]
            else False
        )
        raised_exception = False

    except (ValueError, ToolException) as e:
        print("Error on ", question, e)
        response = {"output": None, "intermediate_steps": None}
        parsing_error = False
        iteration_limit_exceeded = False
        exception = e
        raised_exception = True

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # collect results
    if response["intermediate_steps"] is not None:
        intermediate_steps = [
            {
                "tool": response[0].tool,
                "tool_input": response[0].tool_input,
                "tool_output": response[1],
            }
            for response in response["intermediate_steps"]
        ]
    else:
        intermediate_steps = None
    return {
        "agent_name": agent_name,
        "question": question,
        "prediction": response["output"],
        "intermediate_steps": intermediate_steps,
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": repr(exception) if raised_exception else None,
        "start_time": start_time,
        "end_time": end_time,
    }



def run_agent(
    question: str,
    agent_executor: AgentExecutor,
    agent_name: str,
    agent_call_function: Callable,
) -> dict:
    """
    Runs the execution process for a given question and ground truth answer.

    Args:
        question (str): The input question to be evaluated.
        agent_executor (AgentExecutor): The agent executor object used to run the agent.
        agent_name (str): The name of the agent model.

    Returns:
        dict: A dictionary containing the evaluation results, including the agent model ID, evaluator model ID,
        question, ground truth answer, prediction, intermediate steps, evaluation score, evaluation feedback,
        tool call parsing error flag, iteration limit exceeded flag, and agent error (if any).
    """
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        # run executor agent
        response = agent_call_function(agent_executor, question)

        # check for parsing errors which indicate the LLM failed to follow the ReACT format
        # this could be due to an issue with the tool calling format or ReACT formatting (i.e. Thought, Action, Observation, etc.)
        parsing_error = (
            True
            if any(
                [
                    "Could not parse LLM output" in step[0].log
                    for step in response["intermediate_steps"]
                ]
            )
            else False
        )

        # check if iteration limit exceeded
        iteration_limit_exceeded = (
            True
            if "Agent stopped due to iteration limit or time limit." in response["output"]
            else False
        )
        raised_exception = False

    except Exception as e:
        response = {"output": None, "intermediate_steps": None}
        parsing_error = False
        iteration_limit_exceeded = False
        exception = e
        raised_exception = True

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # collect results
    if response["intermediate_steps"] is not None:
        intermediate_steps = [
            {
                "tool": response[0].tool,
                "tool_input": response[0].tool_input,
                "tool_output": response[1],
            }
            for response in response["intermediate_steps"]
        ]
    else:
        intermediate_steps = None
    return {
        "agent_name": agent_name,
        "question": question,
        "prediction": response["output"],
        "intermediate_steps": intermediate_steps,
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
    output_folder: str = "output",
    agent_call_function: Callable = call_langchain_agent,
    key_for_answer: str = "answer",
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
        with open(f"{output_folder}/{agent_name}.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        results = []

    results_df = pd.DataFrame(results)

    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        if len(results_df) > 0:
            if example["question"] in results_df["question"].unique():
                continue

        # run agent
        result = await arun_agent(
            question=example["question"],
            agent_executor=agent_executor,
            agent_name=agent_name,
            agent_call_function=agent_call_function,
        )

        # add in example metadata
        result.update(
            {
                "gt_answer": example[key_for_answer],
                "task": example["task"],
            }
        )
        results.append(result)

        with open(f"{output_folder}/{agent_name}.json", "w") as f:
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
            agent_call_function=acall_langchain_agent,
        )
        for agent_name, agent_executor in agents.items()
    ]

    results = await asyncio.gather(*tasks)

    return pd.DataFrame([element for sublist in results for element in sublist])
