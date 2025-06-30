from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
# from langgraph.graph import MessagesState, StateGraph
# from langchain_core.messages import SystemMessage
# from langgraph.prebuilt import ToolNode
# from langchain_core.tools import tool
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

# def split_text(docs, chunk_size: int, overlap: int):
#     """Splits documents by specific chunk size and overlap"""
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size, chunk_overlap=overlap
#     )
#     return splitter.split_documents(docs)

# def store_vectors(doc_splits: list):
#     pass

# def build_graph():
#     return StateGraph(MessagesState)

# @tool(response_format="content_and_artifact")
# def retrieve(vector_store, query: str, k: int):
#     """Retrieve information related to a query"""
#     retrieved_docs = vector_store.similarity_search(query, k)
#     serialized = "\n\n".join(
#         (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
#         for doc in retrieved_docs
#     )
#     return serialized, retrieved_docs

# def query_or_respond(llm, state: MessagesState):
#     """Generate tool call for retrieval or respond."""
#     llm_with_tools = llm.bind_tools([retrieve])
#     response = llm_with_tools.invoke(state["messages"])
#     # MessagesState appends messages to state instead of overwriting
#     return {"messages": [response]}