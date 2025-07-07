from langgraph.graph import MessagesState
from utils import create_llm
from retriever import generate_retreiver
# from langchain_core.messages import convert_to_messages

MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.0
response_model = create_llm(MODEL_NAME, TEMPERATURE)

retreiver_tool = generate_retreiver()

rewrite_prompt = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)

generate_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)

def generate_query_or_respond(state: MessagesState):
    """
    Calls the model to generate a response on the current state.
    It will decide to retireve 
    """
    response = (
        response_model.bind_tools([retreiver_tool]).invoke(state["messages"])
    )
    return {"messages": [response]}

def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    question = state["messages"][0].content
    formatted_prompt = rewrite_prompt.format(question=question)
    response = response_model.invoke([{"role": "user", "content": formatted_prompt}])
    return {"messages": [{"role": "user", "content": response.content}]}

def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    formatted_prompt = generate_prompt.format(question=question, context=context)
    response = response_model.invoke({"role": "user", "content": formatted_prompt})
    return {"messages": [response]}