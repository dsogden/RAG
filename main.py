from langchain_tools import create_llm, create_vector_store
from langchain_openai import OpenAIEmbeddings
from env_config import MyConfig
import os

config = MyConfig(
    MODEL=os.getenv("OPENAI_API_MODEL"),
    MODEL_PROVIDER=os.getenv("OPENAI_API_MODEL_PROVIDER"),
    API_KEY=os.getenv("OPENAI_API_KEY"),
    EMBEDDING_MODEL=os.getenv("OPENAI_API_EMBEDDINGS_MODEL")
)

llm = create_llm(config.MODEL, config.MODEL_PROVIDER, config.API_KEY)
embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
vector_store = create_vector_store(embeddings)