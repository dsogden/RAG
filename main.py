from langchain_tools import create_llm, create_vector_store
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-large"
API_KEY = os.getenv('OPENAI_API_KEY')
CHUNK_SIZE = 1000
SPLIT_OVERLAP = 200

llm = create_llm(MODEL_NAME, API_KEY)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vector_store = create_vector_store(embeddings)

# path = "https://lilianweng.github.io/posts/2023-06-23-agent/"
# loader = webpage_loader(path, classes=("post-content", "post-title", "post-header"))
# docs = loader.load()
# all_splits = split_text(docs, CHUNK_SIZE, SPLIT_OVERLAP)
# stored_vectors = vector_store.add_documents(documents=all_splits)
# # serialized, retrieved_docs = retrieve()

# print(llm.invoke(input=[HumanMessage(content="Hello")]))
