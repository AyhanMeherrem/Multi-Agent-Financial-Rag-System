import html
import os
import requests
import streamlit as st

st.set_page_config(page_title="Financial RAG Agent", page_icon="📊", layout="centered")

st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(91,140,255,0.16) 0%, rgba(91,140,255,0) 45%),
            radial-gradient(circle at 85% 0%, rgba(139,92,246,0.14) 0%, rgba(139,92,246,0) 40%),
            linear-gradient(180deg, #0f1117 0%, #151823 100%);
    }
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #2d3348;
        color: #6b7385;
        font-size: 0.85rem;
    }
    .footer a {
        color: #8ba3ff;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    .subtitle {
        color: #9aa4b2;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stButton"] button {
        border: 1px solid #2d3348;
        background-color: #1a1e2b;
        color: #d6dbe6;
        transition: all 0.15s ease-in-out;
    }
    div[data-testid="stButton"] button:hover {
        border-color: #5b8cff;
        color: #ffffff;
        background-color: #222840;
    }
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: #5b8cff;
        border: none;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background-color: #4a7aef;
    }
    .answer-card {
        background-color: #1a1e2b;
        border: 1px solid #2d3348;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Financial Expert Agent")
st.markdown(
    '<div class="subtitle">Ask about AAPL / MSFT 10-K filings — revenue, risk factors, MD&A, and more. '
    'Compare both companies in one question.</div>',
    unsafe_allow_html=True,
)

EXAMPLE_QUERIES = [
    "Compare Apple's and Microsoft's total net revenue for fiscal year 2024.",
    "What were Apple's primary risk factors in 2024?",
    "What were Microsoft's primary cybersecurity risks in 2024?",
]

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

st.markdown("**Try an example**")
example_cols = st.columns(len(EXAMPLE_QUERIES))
for col, example in zip(example_cols, EXAMPLE_QUERIES):
    if col.button(example, use_container_width=True):
        st.session_state.query_input = example

user_query = st.text_input("Query", key="query_input", placeholder="Ask something about AAPL or MSFT's 10-K...")
isGenerateClicked = st.button("Generate", type="primary")

url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/query")
backend_api_key = os.getenv("BACKEND_API_KEY", "")

if isGenerateClicked:
    try:
        with st.spinner("Routing query and retrieving filings..."):
            response = requests.post(
                url,
                json={"query": user_query},
                headers={"X-Internal-Key": backend_api_key},
                timeout=60,
            )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the backend at {url}: {e}")
    else:
        result = response.json()
        # This is rendered via unsafe_allow_html=True, so the LLM's raw output must be
        # HTML-escaped first — otherwise a prompt-injected "<script>...</script>" (or any
        # stray "<"/">"/"&") in the answer would be parsed as live markup, not text.
        # Escape before the "$" fix below: html.escape() would otherwise double-encode
        # the "&" we introduce via "&#36;".
        safe_answer = html.escape(result["answer"])
        # Markdown/HTML treats a single "$" as the start of a LaTeX math span, which
        # mangles dollar amounts like "$391,035 million" — swap in the literal entity.
        safe_answer = safe_answer.replace("$", "&#36;")
        st.markdown(f'<div class="answer-card">{safe_answer}</div>', unsafe_allow_html=True)

        badge_cols = st.columns(2)
        badge_cols[0].metric("Companies", ", ".join(result.get("companies") or []) or "—")
        badge_cols[1].metric("Year", result.get("year") or "—")

        source_urls = result.get("source_urls")
        if source_urls:
            links = " · ".join(
                f'<a href="{html.escape(link)}" target="_blank">{html.escape(company)} 10-K on SEC EDGAR</a>'
                for company, link in source_urls.items()
            )
            st.markdown(f'<div class="subtitle" style="margin-top:0.75rem;">Sources: {links}</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer">Built by <a href="https://github.com/AyhanMeherrem/Multi-Agent-Financial-Rag-System" '
    'target="_blank">Ayhan Meherrem</a></div>',
    unsafe_allow_html=True,
)
