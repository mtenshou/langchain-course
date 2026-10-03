from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama


tavily = TavilyClient


@tool
def search(query: str) -> str:
    """A bare LLM is essentially a chat interface — it receives text and returns text. 
    If you want it to do something (search the web, run code, query a database), you need to equip it with tools: functions the model can choose to call. 
    Writing and wiring these up one by one becomes tedious fast, 
    which is part of what motivated the MCP (Model Context Protocol) standard — a unified way to expose tools to models without reimplementing the plumbing each time.
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)


# llm = ChatOpenAI(model="gpt-5.4-mini")
# 1.75+1.70=3.45s

llm = ChatOllama(model="qwen3.5:0.8b")
# gemma3:270mm is not supported to tool call
# speed gemma4:4b=72s+15s(87s)  gwen3.5:4b=9+33(42s) qwen3.5:0.8b=8.74+18.41(20.15)

tools = [TavilySearch()]
agent = create_agent(model=llm,tools=tools)

def main():
    print("Hello from langchain-agent!")
    response = agent.invoke(
        {"messages":HumanMessage(content="Search for 3 AI engeer jobs in Tokyo"),}
    )
    print(response)



if __name__ == "__main__":
    main()
