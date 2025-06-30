from langchain_tools import Model, TextEmbeddings, create_llm

model = Model()
embedings = TextEmbeddings()

llm = create_llm(model)

print(llm)