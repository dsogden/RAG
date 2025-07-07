from fastapi import FastAPI
from langchain_utils.chatbot import run_chatbot
from pydantic import BaseModel

app = FastAPI(
    title="Baseball chatbot",
    description="Endpoint for a general baseball system graph RAG chatbot"
)

class Query(BaseModel):
    query: str

class Response(BaseModel):
    response: str

@app.post("/baseball_info")
async def post(query: Query) -> Response:
    response = run_chatbot(query.query)
    return {"response": response["messages"][-1].content}