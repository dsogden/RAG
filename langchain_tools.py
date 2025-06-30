from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from env_config import MyConfig

def create_llm(config: MyConfig):
    """Create the llm model"""
    return init_chat_model(
        model=config.MODEL,
        model_provider=config.MODEL_PROVIDER,
        api_key=config.API_KEY
    )

def create_embeddings(config: MyConfig):
    """Create the embedding model"""
    return OpenAIEmbeddings(model=config.TEXT_EMBEDDINGS, api_key=config.API_KEY)

def create_vector_store(embeddings: OpenAIEmbeddings):
    """Create the vector store"""
    return InMemoryVectorStore(embeddings)
