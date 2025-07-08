# A basic LLM RAG chatbot with LangChain

This repo contains the source code for an LLM RAG chatbot, built with LangChain. The goal of the project was to learn to build out a basic implementation of a chatbot that will retrieve some general facts about baseball (i.e. pitching rules, pitch grips, and how to play the game).

If you'd like to download this repo use the following:
```
$ git clone https://github.com/dsogden/RAG.git
```

The chatbot uses OpenAI LLMs, specifically "gpt-4o-mini", and will require an OpenAI API Key. I have also set up tracing with LangSmith so you need an API key for that as well. All of this information needs to be dumped into a .env file in the root directory with the following format:
```
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY=<LangSmith API Key>
LANGSMITH_PROJECT=<LangSmith Project Name>
OPENAI_API_KEY=<OPENAI API Key>
```

# Example Screenshot of the functionality.
The frontend for the implementation is Streamlit.
![screenshot](https://github.com/dsogden/RAG/blob/main/example.png)
