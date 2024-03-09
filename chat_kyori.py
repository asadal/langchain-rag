import streamlit as st
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
# from create_database import generate_data_store
import os

# these three lines swap the stdlib sqlite3 lib with the pysqlite3 package
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
질문에 대한 답변은 다음 컨텍스트를 바탕으로 합니다:

{context}

---

위 컨텍스트를 바탕으로 한 질문의 답변: {question}
"""

def chat_with_db(query_text):
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        return "관련 결과를 찾을 수 없습니다."

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    model = ChatOpenAI()
    response = model.invoke(prompt)  # `invoke` 메서드 사용
    # 응답에서 'content' 키의 값을 반환
    response_text = response.content
    return response_text

def main():# title nad fabicon
    st.set_page_config(page_title = "Hani Kyori Bot", page_icon = "https://img.hani.co.kr/imgdb/original/2024/0116/1817053905854034.png")

    # featured image
    st.image("https://flexible.img.hani.co.kr/flexible/normal/240/240/imgdb/original/2021/0524/20210524502287.jpg", width=100)

    st.title("겨리봇")
    st.markdown("한겨레 후원회원 '서포터즈 벗'을 위한 인공지능 챗봇입니다. 후원회원 관련 궁금한 점을 물어보세요.")

    # generate_data_store()

    if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

    for content in st.session_state.chat_history:
        with st.chat_message(content["role"]):
            st.markdown(content['message'])

    # `st.chat_input`으로 사용자 입력 받기.
    if prompt := st.chat_input("한겨레 후원회원이 뭔가요?"):
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
            st.session_state.chat_history.append({"role": "user", "message": prompt})
    # user_input = st.chat_input("Ask Kyori:", key="chat_input")

        # 챗봇으로부터 응답 받기
        with st.chat_message("Kyori", avatar="https://raw.githubusercontent.com/asadal/langchain-rag/main/images/kyori.png"):
            response = chat_with_db(prompt)
            st.markdown(response)
            st.session_state.chat_history.append({"role": "Kyori", "message": response})
    # if user_input:
    #     response = chat_with_db(user_input)
    #     st.chat_message("User").write(user_input)
    #     st.chat_message("Kyori").write(response)  # 응답 출력

if __name__ == "__main__":
    main()
