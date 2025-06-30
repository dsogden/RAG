from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from dataclasses import dataclass
from env_config import MyConfig

config = MyConfig()

@dataclass
class Model:
    model: str = config.MODEL
    model_provider: str = config.MODEL_PROVIDER

@dataclass
class TextEmbeddings:
    text_embedding: str = config.TEXT_EMBEDDINGS

def create_llm(model_info: Model):
    """Create the llm model"""
    return init_chat_model(
        model=model_info.model,
        model_provider=model_info.model_provider,
        api_key=config.API_KEY
    )

def embed_docments(embedding_model: OpenAIEmbeddings):
    pass