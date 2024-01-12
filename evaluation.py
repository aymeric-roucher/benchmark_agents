import json
from tqdm.notebook import tqdm
from langchain.llms import HuggingFaceEndpoint
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.chat_models import ChatOpenAI


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
            "max_new_tokens": 512,
            "do_sample": False,
            "repetition_penalty": 1.03,
        },
    )
    return eval_chat_model


def evaluate_answers(
    results_path: str,
    evaluator,
    evaluator_name: str,
    eval_prompt_template: ChatPromptTemplate,
) -> None:
    if isinstance(evaluator, HuggingFaceEndpoint):
        evaluate_answers_huggingface(
            results_path, evaluator, evaluator_name, eval_prompt_template
        )
    elif isinstance(evaluator, ChatOpenAI):
        evaluate_answers_openai(
            results_path, evaluator, evaluator_name, eval_prompt_template
        )


def evaluate_answers_huggingface(
    results_path: str,
    evaluator: ChatHuggingFace,
    evaluator_name: str,
    eval_prompt_template: ChatPromptTemplate,
) -> None:
    """
    Runs the evaluation process on a file of results. Used to perform an evaluation with a specific evaluator model.

    Args:
        results_path (str): The path to the results file.
        evaluator (HuggingFaceChatWrapper): The evaluator object used to evaluate the agent's response.
        eval_prompt_template (ChatPromptTemplate): The template for the evaluation prompt.
    """

    with open(results_path, "r") as f:
        results = json.load(f)

    for experiment in tqdm(results):
        if f"eval_score_{evaluator_name}" in experiment:
            continue

        eval_prompt = eval_prompt_template.format_messages(
            instruction=experiment["question"],
            response=experiment["prediction"],
            reference_answer=experiment["gt_answer"],
        )
        eval_result = evaluator.invoke(eval_prompt)
        try:
            feedback, score = [item.strip() for item in eval_result.split("[RESULT]")]
        except:
            # Split the string by '[RESULT]' and filter out empty segments
            segments = [
                segment.strip()
                for segment in eval_result.split("[RESULT]")
                if segment.strip()
            ]

            # Search for a segment that contains a numerical score
            for segment in segments:
                if segment.isdigit():
                    feedback = ""
                    score = int(segment)

        experiment[f"eval_score_{evaluator_name}"] = score
        experiment[f"eval_feedback_{evaluator_name}"] = feedback

        with open(results_path, "w") as f:
            json.dump(results, f)


def evaluate_answers_openai(
    results_path: str,
    evaluator: ChatHuggingFace,
    evaluator_name: str,
    eval_prompt_template: ChatPromptTemplate,
) -> None:
    """
    Runs the evaluation process on a file of results. Used to perform an evaluation with a specific evaluator model.

    Args:
        results_path (str): The path to the results file.
        evaluator (HuggingFaceChatWrapper): The evaluator object used to evaluate the agent's response.
        eval_prompt_template (ChatPromptTemplate): The template for the evaluation prompt.
    """

    with open(results_path, "r") as f:
        results = json.load(f)

    for experiment in tqdm(results):
        if f"eval_score_{evaluator_name}" in experiment:
            continue

        eval_prompt = eval_prompt_template.format_messages(
            instruction=experiment["question"],
            response=experiment["prediction"],
            reference_answer=experiment["gt_answer"],
        )
        eval_result = evaluator.invoke(eval_prompt)
        feedback, score = [
            item.strip() for item in eval_result.content.split("[RESULT]")
        ]
        experiment[f"eval_score_{evaluator_name}"] = score
        experiment[f"eval_feedback_{evaluator_name}"] = feedback

        with open(results_path, "w") as f:
            json.dump(results, f)
