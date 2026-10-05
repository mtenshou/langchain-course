from dotenv import load_dotenv
import os
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader("E:\project\langchain_tutorial\langchain-rag\mediumblog1.txt",
                        encoding="UTF-8"
            )
    document = loader.load()

    print("spritting...")
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created{len(texts)}chunks")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("ingesting...")
    PineconeVectorStore.from_documents(texts,embeddings,index_name=os.environ['INDEX_NAME'])
    print("finish...")