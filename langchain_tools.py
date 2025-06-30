from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

def create_llm(
    model_name: str, model_provider: str, api_key: str
):
    """Create the llm model"""
    return init_chat_model(
        model=model_name,
        model_provider=model_provider,
        api_key=api_key
    )

def create_vector_store(embeddings: OpenAIEmbeddings):
    """Create the vector store"""
    return InMemoryVectorStore(embeddings)




