from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI




def main():
    print("Hello from langchain-agent!")


if __name__ == "__main__":
    main()
