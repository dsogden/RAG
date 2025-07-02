from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from pypdf import PdfReader
# import os

def create_llm(model_name: str):
    """Create the llm model"""
    return ChatOpenAI(model=model_name)

def create_vector_store(embeddings):
    """Create the vector store"""
    return InMemoryVectorStore(embeddings)

# def read_document(document: str):
#     """Extracts text from a pdf file"""
#     try:
#         reader = PdfReader(document)
#         extracted_text = []
#         N = len(reader.pages)
#         for page_num in range(N):
#             page = reader.pages[page_num]
#             extracted_text.append(page.extract_text())
#         return "\n".join(extracted_text)
#     except Exception as e:
#         print(f"Error reading PDF: {e}")
#         return None
    
# def compose_documents(path: str):
#     """Returns a list of extracted documents"""
#     documents = os.listdir(path)
#     return [
#         read_document(os.path.join(path, document))
#         for document in documents
#     ]

# def split_text(docs, chunk_size: int, overlap: int):
#     """Splits documents by specific chunk size and overlap"""
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size, chunk_overlap=overlap
#     )
#     return splitter.split_documents(docs)