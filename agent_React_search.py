import os
from dotenv import load_dotenv

load_dotenv()

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent

from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from langchain_ollama import ChatOllama 
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse




# 1. プロンプトの取得（hubから持ってくるのが現在の主流）
react_prompt = hub.pull("hwchase17/react")


# 2. LLMとツールの準備
tools =[TavilySearch()]
llm_ollama = ChatOllama(model="qwen3.5:0.8b",temperature=0)
llm_anth = ChatAnthropic(model="claude-sonnet-4-5-20250929",temperature=0) # type: ignore

output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
structured_react_prompt = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names"]
).partial(format_instructions=output_parser.get_format_instructions()) # type: ignore


# 3. エージェントの作成  rto build an easonable agent
agent = create_react_agent(llm_anth, tools, structured_react_prompt)

# 4. Executorの作成
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
extract_output = RunnableLambda(lambda x: x["output"])
parse_output = RunnableLambda(lambda x: output_parser.parse(x))
chain = agent_executor | extract_output | parse_output


result = chain.invoke(
        input={
            "input":"search for 2 job posting for an ai engineer using langchain in the tokyo area on linkedin ",

        }
)
print(result)


