from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from agent import generate_query_or_respond, rewrite_question, generate_answer
from retriever import generate_retreiver
from document_grading import grade_documents

retriever_tool = generate_retreiver()
memory = MemorySaver()

def build_state_graph():
    workflow = StateGraph(MessagesState)
    workflow.add_edge(START, "generate_query_or_respond")
    workflow.add_node(generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node(rewrite_question)
    workflow.add_node(generate_answer)
    
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {"tools": "retrieve", END: END}
    )
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents
    )

    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")
    graph = workflow.compile(checkpointer=memory)
    return graph

