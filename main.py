from langchain_tools.langchain_tools import create_llm
from agent import build_agent
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "gpt-4o-mini"
llm = create_llm(MODEL_NAME)

config = {"configurable": {"thread_id": "abc123"}}
input_message = (
    "What is the standard method for Task Decomposition?\n\n"
)
agent_executor = build_agent(llm)

outputs = []
for idx, event in enumerate(agent_executor.stream(
{"messages": [{"role": "user", "content": input_message}]},
stream_mode="values",
config=config,
)):
    outputs.append({idx: event["messages"][-1].content})
print(outputs)