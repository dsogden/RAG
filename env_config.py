from dataclasses import dataclass
import os

@dataclass
class MyConfig:
    API_KEY: str = os.getenv("OPENAI_API_KEY")
    MODEL: str = os.getenv("OPENAI_API_MODEL")
    MODEL_PROVIDER: str = os.getenv("OPENAI_API_MODEL_PROVIDER")
    TEXT_EMBEDDINGS: str = os.getenv("OPENAI_API_TEXT_EMBEDDINGS")