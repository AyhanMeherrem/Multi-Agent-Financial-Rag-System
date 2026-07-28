import requests
import streamlit as st
st.title("Financial Expert Agent")
user_query = st.text_input("Query")
isGenerateClicked = st.button("Generate")

url = "http://127.0.0.1:8000/query"

if isGenerateClicked:
    response = requests.post(url, json = {"query": user_query})
    answer = response.json()["answer"]
    st.text_area("Answer", answer, key="answer", height=300)