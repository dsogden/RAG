from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_llm(model_name: str):
    """Create the llm model"""
    return ChatOpenAI(model=model_name)

def load_documents(path: str):
    return PyPDFDirectoryLoader(path).load()

def create_vector_store(collection_name: str, embeddings):
    """Create the vector store"""
    return Chroma(
    collection_name=collection_name,
    embedding_function=embeddings
)

def split_text(docs, chunk_size: int, overlap: int):
    """Splits documents by specific chunk size and overlap"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    return splitter.split_documents(docs)