import os
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
# from tavily import TavilyClient

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
agent = create_agent(model=llm_ollama,tools=tools)  

result = agent.invoke({"messages":HumanMessage(content="what is the weather in Tokyo")})
print(result)

    

# agent= llm+ tool

