import html
import logging
import os
import requests
import streamlit as st

logger = logging.getLogger(__name__)

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

user_query = st.text_input("Query", key="query_input", max_chars=500,
                           placeholder="Ask something about AAPL or MSFT's 10-K...")
isGenerateClicked = st.button("Generate", type="primary")
st.caption("⏱️ Runs on a serverless backend that sleeps when idle to save cost  if this is the first "
           "visit in a while, the first response can take up to a minute or two while it wakes up. "
           "That's infrastructure cold start, not the model itself being slow.")

url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/query")
backend_api_key = os.getenv("BACKEND_API_KEY", "")

# The backend only sees this container's IP, so pass on the end user's IP for its per-user rate
# limit. Azure's ingress appends the client IP as the last X-Forwarded-For entry; earlier entries
# can be set by the client itself, so only the last one is used.
def end_user_ip() -> str | None:
    forwarded_for = st.context.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",")[-1].strip()
    return st.context.ip_address


ERROR_MESSAGES = {
    429: "Too many requests. Please wait a minute and try again.",
    422: "The question must be between 1 and 500 characters.",
}

if isGenerateClicked and not user_query.strip():
    st.warning("Please enter a question first.")
elif isGenerateClicked:
    headers = {"X-Internal-Key": backend_api_key}
    client_ip = end_user_ip()
    if client_ip:
        headers["X-End-User-IP"] = client_ip
    try:
        with st.spinner("Routing query and retrieving filings... (first request after inactivity may take a minute or two, the backend is waking up)"):
            response = requests.post(url, json={"query": user_query}, headers=headers, timeout=180)
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        logger.warning("Backend returned HTTP %s", response.status_code)
        st.error(ERROR_MESSAGES.get(response.status_code, "Something went wrong while generating the answer. Please try again."))
    except requests.exceptions.RequestException:
        # Details (including the backend URL) go to the server log, not to the user
        logger.exception("Could not reach the backend")
        st.error("The service is not reachable right now. Please try again in a minute.")
    else:
        result = response.json()
        # Rendered as Markdown, not raw HTML, so the answer's lists and bold text display properly
        # while any HTML in it (e.g. a prompt-injected <script>) is shown as text, never executed.
        # A bare "$" starts a LaTeX math span in Streamlit, so escape it.
        with st.container(border=True):
            st.markdown(result["answer"].replace("$", "\\$"))

        badge_cols = st.columns(2)
        badge_cols[0].metric("Companies", ", ".join(result.get("companies") or []) or "—")
        badge_cols[1].metric("Year", ", ".join(result.get("years") or []) or result.get("year") or "—")

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
