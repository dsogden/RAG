from langchain_tools import create_llm, create_vector_store, webpage_loader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-large"
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
CHUNK_SIZE = 1000
SPLIT_OVERLAP = 200

llm = create_llm(MODEL_NAME, OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vector_store = create_vector_store(embeddings)
path = "https://lilianweng.github.io/posts/2023-06-23-agent/"
loader = webpage_loader(path, classes=("post-content", "post-title", "post-header"))
docs = loader.load()