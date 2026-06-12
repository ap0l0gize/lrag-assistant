import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_neo4j import Neo4jGraph
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    temperature=0,
    model=os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URL", "bolt://localhost:7687"),
    username=os.getenv("NEO4J_USERNAME", "neo4j"),
    password=os.getenv("NEO4J_PASSWORD", ""),
    refresh_schema=False,
)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    collection_name="pdfs",
    embedding_function=embeddings,
    persist_directory=os.getenv("DB_PATH", "./db"),
)
