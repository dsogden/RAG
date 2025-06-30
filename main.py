from langchain_tools import create_llm, create_embedding_model
from env_config import MyConfig

config = MyConfig()
llm = create_llm(config)
embeddings = create_embedding_model(config)
print(embeddings)