from utils import create_vector_store
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings

PATH = "../documents/"
EMBEDDING_MODEL = "text-embedding-3-large"

loader = PyPDFDirectoryLoader(PATH)
docs = loader.load()
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vector_store = create_vector_store(embeddings)
vector_store.add_documents(docs)

# @tool(response_format="content_and_artifact")
# def retrieve(query: str, k: int=2) -> tuple:
#     """Retrieve information related to query."""
#     retrieved_docs = vectors_store.similarity_search(query, k)
#     serialized = "\n\n".join(
#         (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
#         for doc in retrieved_docs
#     )
#     return serialized, retrieved_docs