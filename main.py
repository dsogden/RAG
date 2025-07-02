from langchain_tools.utils import compose_documents, create_vector_store
import os
# from agent import build_agent
from dotenv import load_dotenv

load_dotenv()

# PATH = "./documents/"
# print(compose_documents(PATH))

# MODEL_NAME = "gpt-4o-mini"
# llm = create_llm(MODEL_NAME)

# config = {"configurable": {"thread_id": "abc123"}}
# input_message = (
#     "What is the standard method for Task Decomposition?\n\n"
# )

# agent_executor = build_agent(llm)

# outputs = []
# for idx, event in enumerate(agent_executor.stream(
# {"messages": [{"role": "user", "content": input_message}]},
# stream_mode="values",
# config=config,
# )):
#     outputs.append({idx: event["messages"][-1].content})
# print(outputs)