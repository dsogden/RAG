from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
import bs4

def create_llm(
    model_name: str, api_key: str
):
    """Create the llm model"""
    return ChatOpenAI(model=model_name, api_key=api_key)

def create_vector_store(embeddings):
    """Create the vector store"""
    return InMemoryVectorStore(embeddings)

def webpage_loader(path: str, classes: tuple[str]):
    """Loads webpage"""
    return WebBaseLoader(
        web_path=path,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=classes
            )
        ), 
        
    )

def split_text(docs, chunk_size: int, overlap: int):
    """Splits documents by specific chunk size and overlap"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    return splitter.split_documents(docs)