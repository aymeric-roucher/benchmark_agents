from typing import List, Optional


from langchain.agents.output_parsers import (
    ReActJsonSingleInputOutputParser,
    OpenAIFunctionsAgentOutputParser,
)
from langchain.llms import HuggingFaceEndpoint
from langchain.chat_models import ChatOpenAI
from langchain.tools.render import (
    render_text_description_and_args,
    format_tool_to_openai_function,
)
from langchain.agents.format_scratchpad import (
    format_to_openai_function_messages,
    format_log_to_str,
)
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.tools import WikipediaQueryRun, tool
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import AgentExecutor, load_tools
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)

from jinja2.exceptions import TemplateError

from langchain.chat_models.base import BaseChatModel
from langchain_community.chat_models.huggingface import ChatHuggingFace

from prompts import HUMAN_PROMPT, SYSTEM_PROMPT

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


@tool
def search_wikipedia(query: str) -> str:
    """Searches Wikipedia for a query. This will not be relevant for the latest information, but it can be useful for historical knowledge."""
    return wikipedia.run(query)


def init_tools_with_llm(llm: BaseChatModel) -> List[tool]:
    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    # Rename tools in the same format used by other tools
    tools[0].name = "search"
    tools[1].name = "calculator"
    # tools.append(search_wikipedia)
    return tools


def build_openai_agent(model_id: Optional[str] = "gpt-4-1106-preview"):
    llm = ChatOpenAI(model=model_id, temperature=0)
    tools = init_tools_with_llm(llm)
    llm_with_tools = llm.bind(
        functions=[format_tool_to_openai_function(t) for t in tools]
    )
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


def build_hf_agent(hf_endpoint_url: str, no_system_prompt=False):
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

    chat_model = ChatHuggingFace(llm=llm)
    tools = init_tools_with_llm(llm)

    # define the prompt depending on whether the chat model supports system prompts
    system_prompt_supported = check_supports_system_prompt(chat_model)

    if system_prompt_supported:
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
                HumanMessagePromptTemplate.from_template(HUMAN_PROMPT),
            ]
        )
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate.from_template(
                    SYSTEM_PROMPT + "\nSo, here is my question:" + HUMAN_PROMPT
                ),
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


def check_supports_system_prompt(chat_model):
    """
    Checks if the given chat model supports system prompts.

    Args:
        chat_model: The chat model to be checked.

    Returns:
        True if the chat model supports system prompts, False otherwise.
    """
    messages = [
        SystemMessage(content="You're a helpful assistant"),
        HumanMessage(
            content="What happens when an unstoppable force meets an immovable object?"
        ),
    ]
    try:
        chat_model._to_chat_prompt(messages)
        return True
    except TemplateError:
        return False
