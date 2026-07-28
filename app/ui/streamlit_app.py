import requests
import streamlit as st
st.title("Financial Expert Agent")
user_query = st.text_input("Query")
isGenerateClicked = st.button("Generate")

url = "http://127.0.0.1:8000/query"

if isGenerateClicked:
    try:
        with st.spinner("Thinking..."):
            response = requests.post(url, json={"query": user_query}, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the backend at {url}: {e}")
    else:
        result = response.json()
        st.text_area("Answer", result["answer"], key="answer", height=300)
        st.caption(f"Companies: {result.get('companies')}  |  Year: {result.get('year')}")