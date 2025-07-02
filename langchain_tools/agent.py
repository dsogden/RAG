from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_tools.retriever import retrieve

memory = MemorySaver()

def build_agent(llm):
    return create_react_agent(
        llm, [retrieve], checkpointer=memory
    )