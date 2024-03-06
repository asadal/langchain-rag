import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import GPT4AllEmbeddings  # 수정된 임포트 경로
from langchain_core.prompts import ChatPromptTemplate
import ollama

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
질문에 대한 답변은 다음 컨텍스트를 바탕으로 합니다:

{context}

---

위 컨텍스트를 바탕으로 한 질문의 답변: {question}
"""

def chat_with_ollama(query_text):
    # GPT4AllEmbeddings 또는 다른 적절한 차원의 임베딩 함수 사용
    embedding_function = GPT4AllEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    # 데이터베이스와 일치하는 차원의 임베딩 모델 사용 확인
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        return "관련 결과를 찾을 수 없습니다."

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Ollama 및 ChatOllama 초기화
    # llm = Ollama(model="llama2")  # 모델 식별자 확인 필요
    model = ChatOllama(model="gemma:2b")  # ChatOllama의 정확한 초기화 방법 확인 필요
    response_text = model.invoke(prompt)
    return response_text

def main():
    st.title("Chat with Kyori")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for content in st.session_state.chat_history:
        with st.chat_message(content["role"]):
            st.markdown(content['message'])

    prompt = st.chat_input("Ask Kyori")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "message": prompt})
        response = chat_with_ollama(prompt)
        st.session_state.chat_history.append({"role": "Kyori", "message": response})

        # Refresh chat history display
        st.rerun()

if __name__ == "__main__":
    main()
