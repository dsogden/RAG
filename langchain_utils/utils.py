from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_llm(model_name: str, temperature: float):
    """Create the llm model"""
    return ChatOpenAI(model=model_name, temperature=temperature)

def load_documents(path: str):
    """Load the documents"""
    return PyPDFDirectoryLoader(path).load()

def create_vector_store(doc_splits: list, embeddings):
    """Create the vector store"""
    return Chroma.from_documents(
        documents=doc_splits,
        embedding=embeddings
    )

def split_text(docs, chunk_size: int, overlap: int):
    """Splits documents by specific chunk size and overlap"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    return splitter.split_documents(docs)