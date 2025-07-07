from fastapi import FastAPI

from langchain_utils.chatbot import run_chatbot

app = FastAPI(
    title="Baseball chatbot",
    description="Endpoint for a general baseball system graph RAG chatbot"
)

query = "What is the grip for a 4 seam fastball?"
response = run_chatbot(query)
@app.get("/")
async def run():
    return {"response": response["messages"][-1].content}