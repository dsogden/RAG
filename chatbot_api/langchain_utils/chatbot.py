from langchain_utils.state_graph import build_state_graph
from langchain_core.messages import HumanMessage

graph = build_state_graph()
config = {"configurable": {"thread_id": "abc123"}}

def run_chatbot(message: str):
    input_message = {"messages": [HumanMessage(message)]}
    result = graph.invoke(input=input_message, config=config)
    return result