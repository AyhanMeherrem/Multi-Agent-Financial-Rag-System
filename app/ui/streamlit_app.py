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
    .chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 0.25rem 0 0.5rem 0;
    }
    .chip {
        border: 1px solid #2d3348;
        background-color: #1a1e2b;
        border-radius: 999px;
        padding: 0.25rem 0.75rem;
        font-size: 0.85rem;
        color: #d6dbe6;
    }
    .chip b {
        color: #8ba3ff;
        margin-right: 0.35rem;
    }
    .chip span {
        color: #6b7385;
        margin-left: 0.35rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Financial Expert Agent")
st.markdown(
    '<div class="subtitle">Ask about the 10-K filings of large US tech companies: revenue, risk factors, '
    'MD&A and more, with every figure cited to its filing. Compare up to three companies in one question.</div>',
    unsafe_allow_html=True,
)

url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/query")
catalog_url = url.rsplit("/query", 1)[0] + "/catalog"
backend_api_key = os.getenv("BACKEND_API_KEY", "")

# The backend only sees this container's IP, so pass on the end user's IP for its per-user rate
# limit. Azure's ingress appends the client IP as the last X-Forwarded-For entry; earlier entries
# can be set by the client itself, so only the last one is used.
def end_user_ip() -> str | None:
    forwarded_for = st.context.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",")[-1].strip()
    return st.context.ip_address


def backend_headers() -> dict:
    headers = {"X-Internal-Key": backend_api_key}
    client_ip = end_user_ip()
    if client_ip:
        headers["X-End-User-IP"] = client_ip
    return headers


# Shown while the backend is asleep or unreachable; the live list comes from /catalog
FALLBACK_CATALOG = {"companies": [{"ticker": t, "name": n, "years": ["2024", "2025"]} for t, n in [
    ("AAPL", "Apple"), ("AMZN", "Amazon"), ("GOOGL", "Alphabet"), ("META", "Meta Platforms"),
    ("MSFT", "Microsoft"), ("NVDA", "NVIDIA")]], "max_companies_per_question": 3}


# The index only changes with a new deployment, so one successful answer is kept for an hour. A
# failure raises and is not cached. The short timeout keeps a sleeping backend from holding up the
# page (the request still wakes it up for the first question).
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_catalog() -> dict:
    response = requests.get(catalog_url, headers={"X-Internal-Key": backend_api_key}, timeout=3)
    response.raise_for_status()
    return response.json()


try:
    available = fetch_catalog()
except requests.exceptions.RequestException:
    available = FALLBACK_CATALOG


def years_label(years: list) -> str:
    if len(years) > 1:
        return f"FY{years[0]}–{years[-1][-2:]}"
    return f"FY{years[0]}" if years else ""


chips = "".join(
    f'<div class="chip"><b>{html.escape(c["ticker"])}</b>{html.escape(c["name"])}'
    f'<span>{html.escape(years_label(c["years"]))}</span></div>'
    for c in available["companies"]
)
st.markdown("**Available filings**")
st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)

EXAMPLE_QUERIES = [
    "Compare NVIDIA's and Meta's net income for fiscal year 2025.",
    "How did Amazon's net income change from 2024 to 2025?",
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
                           placeholder="Ask about one to three of the companies above...")
isGenerateClicked = st.button("Generate", type="primary")
st.caption("⏱️ Runs on a serverless backend that sleeps when idle to save cost  if this is the first "
           "visit in a while, the first response can take up to a minute or two while it wakes up. "
           "That's infrastructure cold start, not the model itself being slow.")

ERROR_MESSAGES = {
    429: "Too many requests. Please wait a minute and try again.",
    503: "The language model service is unavailable right now. Please try again shortly.",
    422: "The question must be between 1 and 500 characters.",
}

if isGenerateClicked and not user_query.strip():
    st.warning("Please enter a question first.")
elif isGenerateClicked:
    try:
        with st.spinner("Routing query and retrieving filings... (first request after inactivity may take a minute or two, the backend is waking up)"):
            response = requests.post(url, json={"query": user_query}, headers=backend_headers(), timeout=180)
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

        searched = [", ".join(result.get("companies") or []),
                    ", ".join(f"FY{y}" for y in result.get("years") or [])]
        details = " · ".join(filter(None, searched))
        if result.get("cached"):
            details += " · ⚡ answered from cache"
        if details:
            st.caption(details)

        # The filing excerpts the answer cites, each linked to the filing document on sec.gov
        sources = result.get("sources") or []
        if sources:
            st.markdown("**Sources**")
            for source in sources:
                with st.container(border=True):
                    label = html.escape(f"{source['company']} FY{source['year']}, {source['section']}")
                    link = f' · <a href="{html.escape(source["url"])}" target="_blank">10-K filing</a>' if source.get("url") else ""
                    st.markdown(f"<b>{label}</b>{link}", unsafe_allow_html=True)
                    st.caption(source["snippet"].replace("$", "\\$") + "…")

st.markdown(
    '<div class="footer">Built by <a href="https://github.com/AyhanMeherrem/Multi-Agent-Financial-Rag-System" '
    'target="_blank">Ayhan Meherrem</a> · Based on public SEC filings. Not investment advice.</div>',
    unsafe_allow_html=True,
)
