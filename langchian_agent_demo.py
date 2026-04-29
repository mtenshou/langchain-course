import os
from dotenv import load_dotenv

load_dotenv()

from typing import List
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
# from tavily import TavilyClient

class Source(BaseModel):
    """Scheme for a source used by the agent"""

    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer:str = Field(description="The agent`s to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")


llm_anth= ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0) # type: ignore

# tavily = TavilyClient()
# @tool
# def search(query:str) -> str:
#     """
#     Tool that searches over internet
#     Args:
#         query : The query to search for
#     Returns:
#         The search result
#     """
#     print(f"searching for {query}")
#     return tavily.search(query=query)

llm_ollama = ChatOllama(model="qwen3.5:0.8b",temperature=0)

tools = [TavilySearch()]
agent = create_agent(model=llm_ollama,tools=tools,response_format=AgentResponse)  

result = agent.invoke({"messages":HumanMessage(content="what is the weather in Nagano")}) # type: ignore
print(result)

    

# agent= llm + tool

