import streamlit as st
from lang_utils.chatbot import run_chatbot

st.title("Chat Model with RAG")
st.session_state.messages = []

def main():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask anything"):
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message('user'):
            st.markdown(prompt)
            input_message = prompt

        with st.chat_message('assistant'):
            output = run_chatbot(input_message)
            response = output['messages'][-1].content
            
            st.markdown(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )