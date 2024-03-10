import streamlit as st
from langchain_community.vectorstores import Chroma
# from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.embeddings import GPT4AllEmbeddings  # 수정된 임포트 경로
from langchain_core.prompts import ChatPromptTemplate
# import ollama

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
질문에 대한 답변은 다음 컨텍스트를 바탕으로 합니다:

{context}

---

위 컨텍스트를 바탕으로 한 질문의 답변: {question}
"""

def chat_with_ollama(query_text):
    # GPT4AllEmbeddings 또는 다른 적절한 차원의 임베딩 함수 사용
    # embedding_function = GPT4AllEmbeddings()
    embedding_function = OllamaEmbeddings()
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
    return response_text.content

def main():
    # 타이틀과 파비콘 설정
    st.set_page_config(page_title = "Hani Kyori Bot", page_icon = "https://img.hani.co.kr/imgdb/original/2024/0116/1817053905854034.png")

    # 특징 이미지
    st.image("https://flexible.img.hani.co.kr/flexible/normal/240/240/imgdb/original/2021/0524/20210524502287.jpg", width=100)

    # 제목 표시
    st.title("겨리봇")
    st.markdown("한겨레 후원회원 '서포터즈 벗'을 위한 인공지능 챗봇입니다. 후원회원 관련 궁금한 점을 물어보세요.")

    # 세션 상태에 'chat_history'가 없으면 초기화
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Kyori와 사용자 아바타 URL
    kyori_avatar = "https://raw.githubusercontent.com/asadal/langchain-rag/main/images/kyori.png"
    user01_avatar = "https://raw.githubusercontent.com/asadal/langchain-rag/main/images/user_01.png"

    # 대화 이력을 반복하여 표시하고 각 메시지에 아바타를 포함시킵니다.
    for content in st.session_state.chat_history:
        role = content["role"]
        avatar = content["avatar"] if "avatar" in content else None
        with st.chat_message(role, avatar=avatar):
            st.markdown(content['message'])

    # 사용자 입력 받기
    if prompt := st.chat_input("한겨레 후원회원이 뭔가요?"):
        # 사용자 메시지를 대화 이력에 추가, user01_avatar URL 포함
        st.session_state.chat_history.append({"role": "user", "message": prompt, "avatar": user01_avatar})
        response = chat_with_ollama(prompt)

        # 챗봇으로부터 응답 받기
        # Kyori의 메시지를 대화 이력에 추가, kyori_avatar URL 포함
        st.session_state.chat_history.append({"role": "Kyori", "message": response, "avatar": kyori_avatar})

    # user, Kyori 아바타와 함께 출력
    for content in st.session_state.chat_history:
        role = content["role"]
        avatar = content["avatar"] if "avatar" in content else None
        with st.chat_message(role, avatar=avatar):
            st.markdown(content['message'])

# 메인 함수 실행
if __name__ == "__main__":
    main()
