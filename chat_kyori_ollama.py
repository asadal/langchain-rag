import streamlit as st
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings  # Consider removing if not used elsewhere.
from langchain.prompts import ChatPromptTemplate  # Consider adjusting usage.
import ollama

CHROMA_PATH = "chroma"

# Since we're now using ollama, adjust usage of templates and embeddings as necessary.

def chat_with_ollama(query_text):
    # Use ollama for chat response. Adjust according to your context handling needs.
    response = ollama.chat(model='gemma:7B', messages=[
      {
        'role': 'user',
        'content': query_text,
      },
    ])
    return response['message']['content']

def main():
    st.title("Chat with Kyori")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for content in st.session_state.chat_history:
        with st.chat_message(content["role"]):
            st.markdown(content['message'])

    # `st.chat_input`으로 사용자 입력 받기.
    prompt = st.chat_input("Ask Kyori")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
            st.session_state.chat_history.append({"role": "user", "message": prompt})

        # Get response from ollama instead of the previous chat_with_db function
        response = chat_with_ollama(prompt)
        with st.chat_message("Kyori"):
            st.markdown(response)
            st.session_state.chat_history.append({"role": "Kyori", "message": response})

if __name__ == "__main__":
    main()
