from langchain_tools import Model, create_llm

model_type = "gpt-4o-mini"
model_provider = "openai"
model = Model(
    model=model_type, model_provider=model_provider
)

llm = create_llm(model)

print(llm)