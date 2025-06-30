from langchain.chat_models import init_chat_model
from dataclasses import dataclass
import os

api_key = os.getenv("OPENAI_API_KEY")

@dataclass
class Model:
    model: str
    model_provider: str

def create_llm(model_info: Model) -> None:
    """Create the llm model"""
    return init_chat_model(
        model=model_info.model,
        model_provider=model_info.model_provider,
        api_key=api_key
    )
    