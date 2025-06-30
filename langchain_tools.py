from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langgraph.graph import MessagesState, StateGraph
import bs4

def create_llm(
    model_name: str, model_provider: str
):
    """Create the llm model"""
    return init_chat_model(
        model=model_name,
        model_provider=model_provider,
    )

def create_vector_store(embeddings: OpenAIEmbeddings):
    """Create the vector store"""
    return InMemoryVectorStore(embeddings)

def webpage_loader(path: str, classes: tuple[str]):
    return WebBaseLoader(
        web_path=path,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=classes
            )
        )
    )

def build_graph():
    pass
