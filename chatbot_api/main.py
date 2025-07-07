from fastapi import FastAPI
from langchain_utils.chatbot import run_chatbot

app = FastAPI(
    title="Baseball chatbot",
    description="Endpoint for a general baseball system graph RAG chatbot"
)

@app.get("/")
async def run(query: str):
    response = run_chatbot(query)
    return {"response": response["messages"][-1].content}