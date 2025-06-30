from langchain_tools import create_llm, create_vector_store, webpage_loader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "gpt-4o-mini"
MODEL_PROVIDER = "openai"
EMBEDDING_MODEL = "text-embedding-3-large"

llm = create_llm(MODEL_NAME, MODEL_PROVIDER)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vector_store = create_vector_store(embeddings)

path = "https://lilianweng.github.io/posts/2023-06-23-agent/"
loader = webpage_loader(path, ("post-content", "post-title", "post-header"))
docs = loader.load()
print(docs)