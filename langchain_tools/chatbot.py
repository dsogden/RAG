from state_graph import build_state_graph

graph = build_state_graph()
config = {"configurable": {"thread_id": "abc123"}}
input_message = "How many innings are in baseball?"
messages = {
    "messages": [{"role": "user", "content": input_message}]
}
for step in graph.stream(
    messages, stream_mode="values", config=config
):
    print(step["messages"][-1])