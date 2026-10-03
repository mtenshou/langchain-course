from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


@tool
def search(query: str) -> str:
    """A bare LLM is essentially a chat interface — it receives text and returns text. 
    If you want it to do something (search the web, run code, query a database), you need to equip it with tools: functions the model can choose to call. 
    Writing and wiring these up one by one becomes tedious fast, 
    which is part of what motivated the MCP (Model Context Protocol) standard — a unified way to expose tools to models without reimplementing the plumbing each time.
    """
    print(f"Searching for {query}")
    return "Tokyo weather is sunny"


llm = ChatOpenAI()
tools = [search]
agent = create_agent(model=llm,tools=tools)

def main():
    print("Hello from langchain-agent!")
    response = agent.invoke(
        {"messages":HumanMessage(content="What is the weather in Tokyo"),}
    )
    print(response)



if __name__ == "__main__":
    main()
