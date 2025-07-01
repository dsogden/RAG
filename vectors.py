from dataclasses import dataclass
from langchain_tools import create_vector_store, webpage_loader, split_text
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
EMBEDDING_MODEL = "text-embedding-3-large"
CHUNK_SIZE = 1000
SPLIT_OVERLAP = 200

path = "https://lilianweng.github.io/posts/2023-06-23-agent/"
loader = webpage_loader(path, classes=("post-content", "post-title", "post-header"))
docs = loader.load()
doc_splits = split_text(docs, CHUNK_SIZE, SPLIT_OVERLAP)

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vectors_store = create_vector_store(embeddings)
_ = vectors_store.add_documents(documents=doc_splits)

@tool(response_format="content_and_artifact")
def retrieve(query: str, k: int=2) -> tuple:
    """Retrieve information related to query."""
    retrieved_docs = vectors_store.similarity_search(query, k)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs