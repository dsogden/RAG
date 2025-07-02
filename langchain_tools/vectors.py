from utils import load_documents, create_vector_store, split_text
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

PATH = "../documents/"
EMBEDDING_MODEL = "text-embedding-3-large"
CHUNK_SIZE = 1000
OVERLAP = 200
COLLECTION_NAME = 'baseball_info'

docs = load_documents(PATH)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vector_store = create_vector_store(
    COLLECTION_NAME, embeddings
)
splits = split_text(docs, 1000, 200)
vector_store.add_documents(splits)

@tool(response_format="content_and_artifact")
def retrieve(query: str, k: int=2) -> tuple:
    """Retrieve information related to query."""
    retrieved_docs = vector_store.similarity_search(query, k)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs