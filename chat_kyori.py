import streamlit as st
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
import sys

# stdlib sqlite3 라이브러리를 pysqlite3 패키지로 교체
__import__('pysqlite3')
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
    return response.content  # 'content' 키의 값을 반환

def main():
    st.set_page_config(page_title="Hani Kyori Bot", page_icon="https://img.hani.co.kr/imgdb/original/2024/0116/1817053905854034.png")
    st.image("https://flexible.img.hani.co.kr/flexible/normal/240/240/imgdb/original/2021/0524/20210524502287.jpg", width=100)
    st.title("겨리봇")
    st.markdown("한겨레 후원회원 '서포터즈 벗'을 위한 인공지능 챗봇입니다. 후원회원 관련 궁금한 점을 물어보세요.")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    kyori_avatar = "https://raw.githubusercontent.com/asadal/langchain-rag/main/images/kyori.png"
    user01_avatar = "https://raw.githubusercontent.com/asadal/langchain-rag/main/images/user_01.png"

    # 사용자 입력 받기
    if prompt := st.chat_input("한겨레 후원회원이 뭔가요?"):
        # 사용자 메시지를 대화 이력에 추가
        st.session_state.chat_history.append({"role": "user", "message": prompt, "avatar": user01_avatar})
        response = chat_with_db(prompt)
        # 챗봇 응답을 대화 이력에 추가
        st.session_state.chat_history.append({"role": "Kyori", "message": response, "avatar": kyori_avatar})

        # 대화 이력을 업데이트하고 출력하는 부분
        for content in st.session_state.chat_history:
            role = content["role"]
            avatar = content["avatar"]
            with st.chat_message(role, avatar=avatar):
                st.markdown(content['message'])

if __name__ == "__main__":
    main()
