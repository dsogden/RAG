from langchain_utils.utils import load_documents, create_vector_store, split_text
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import OpenAIEmbeddings

PATH = "../documents/"
EMBEDDING_MODEL = "text-embedding-3-large"
CHUNK_SIZE = 1000
OVERLAP = 200
COLLECTION_NAME = 'baseball_info'
K = 2

# load the documents
docs = load_documents(PATH)

# initialize the embedding model and vector storing
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
splits = split_text(docs, CHUNK_SIZE, OVERLAP)
vector_store = create_vector_store(splits, embeddings)

# initialize retriever and retriever tool
def generate_retreiver():
    retriever = vector_store.as_retriever(k=K)
    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="Retrieve_baseball_info",
        description="Searches and returns info about baseball."
    )
    return retriever_tool