from pydantic import BaseModel

class MyConfig(BaseModel):
    API_KEY: str
    MODEL: str
    MODEL_PROVIDER: str
    EMBEDDING_MODEL: str