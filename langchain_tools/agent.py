from langgraph.graph import MessagesState
from utils import create_llm
from retriever import generate_retreiver

MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.3
llm = create_llm(MODEL_NAME, TEMPERATURE)

retreiver_tool = generate_retreiver()

def generate_query_or_respond(state: MessagesState):
    """
    Calls the model to generate a response on the current state.
    It will decide to retireve 
    """
    response = (
        llm.bind_tools([retreiver_tool]).invoke(state["messages"])
    )
    return {"messages": [response]}