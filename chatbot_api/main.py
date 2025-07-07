from fastapi import FastAPI
from langchain_utils.chatbot import run_chatbot
from pydantic import BaseModel

app = FastAPI(
    title="Baseball chatbot",
    description="Endpoint for a general baseball system graph RAG chatbot"
)

query = "How many innings are in baseball?"
response = run_chatbot(query)

class Response(BaseModel):
    response: str

@app.get("/")
async def run() -> Response:
    output = response["messages"][-1].content
    return {"response": output}