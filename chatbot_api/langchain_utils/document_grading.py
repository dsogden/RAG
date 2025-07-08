from pydantic import BaseModel, Field
from typing import Literal
from langgraph.graph import MessagesState
from langchain_utils.utils import create_llm
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("CHATGPT_MODEL")
TEMPERATURE = os.getenv("TEMPERATURE")

grader_model = create_llm(MODEL_NAME, TEMPERATURE)

grade_prompt = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score : str = Field(description="Releveance score")

def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """Determines whether the retrieved documents are relevant to question"""
    question = state["messages"][0].content
    context = state["messages"][1].content
    formatted_prompt = grade_prompt.format(question=question, context=context)
    response = grader_model.with_structured_output(GradeDocuments).invoke(
        [{"role": "user","content": formatted_prompt}]
    )
    score = response.binary_score
    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"