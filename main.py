import streamlit as st
from langchain_community.llms import CTransformers

llm = CTransformers(
    model="llama-2-7b-chat.ggmlv3.q2_K.bin",
    model_type="llama"
)

st.title('AI 시인')
content = st.text_input('시의 주제를 입력하세요')

if st.button('시 작성'):
    with st.spinner('AI 요청 중'):
        result = llm.invoke("write a poem about : " + content)
        st.write(result)