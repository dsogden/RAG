from langchain_tools import create_llm
from agent import build_agent
from dotenv import load_dotenv
from fastapi import FastAPI
import os

load_dotenv()

app = FastAPI()

MODEL_NAME = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

llm = create_llm(MODEL_NAME, OPENAI_API_KEY)

config = {"configurable": {"thread_id": "abc123"}}
input_message = (
    "What is the standard method for Task Decomposition?\n\n"
    # "Once you get the answer, look up common extensions of that method."
)
agent_executor = build_agent(llm)

@app.get("/")
async def root():
    outputs = []
    for idx, event in enumerate(agent_executor.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
    )):
        outputs.append({idx: event["messages"][-1].content})
    return {"outputs": outputs}