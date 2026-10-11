from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_ollama import ChatOllama

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system"
            "Your are a viral twitter influencer gradding a tweet. Generate critique and recommandations for the user"
            "Always provide detailed recommendations, including requests for length, virality, style, etc"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generatino_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system"
            "You are a twitter techie influencer assistant tasked with writting excellent twitter posts"
            "Generate the best twitter post possible for the user's request"
            "If the user provides critique, respond with a revised version of your previous attempts"
        ),
        MessagesPlaceholder(variable_name="messages",)
    ]
)

llm_ollama = ChatOllama(model="phi4-mini:latest", temperature=0)
generate_chain = generatino_prompt | llm_ollama
reflection_chain = reflection_prompt | llm_ollama