import html
import logging
import os
import re
import requests
import streamlit as st

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Financial RAG on SEC 10-K Filings", page_icon="📊", layout="wide")

REPO_URL = "https://github.com/AyhanMeherrem/Multi-Agent-Financial-Rag-System"
GITHUB_PROFILE_URL = "https://github.com/AyhanMeherrem"
LINKEDIN_URL = "https://www.linkedin.com/in/ayhan-meherrem/"
GITHUB_ICON = ('<svg width="20" height="20" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 '
               '3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23'
               '-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87'
               '.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64'
               '-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 '
               '2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 '
               '0 0016 8c0-4.42-3.58-8-8-8z"/></svg>')
LINKEDIN_ICON = ('<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><rect x="2" y="2" width="20" '
                 'height="20" rx="4"/><text x="12" y="17" text-anchor="middle" font-family="Arial, sans-serif" '
                 'font-size="12" font-weight="700" fill="#ffffff">in</text></svg>')

# Logo: rising bars inside a rounded square with a search dot, drawn inline so there is no image file
LOGO_SVG = """<svg width="30" height="30" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect x="1" y="1" width="30" height="30" rx="8" fill="#0b1b3a"/>
<rect x="8" y="17" width="4" height="8" rx="1.5" fill="#34d399"/>
<rect x="14" y="12" width="4" height="13" rx="1.5" fill="#34d399" opacity="0.8"/>
<rect x="20" y="7" width="4" height="18" rx="1.5" fill="#34d399" opacity="0.6"/>
<circle cx="24" cy="8" r="2.5" fill="#ffffff"/></svg>"""

# Line icons for the feature cards (24x24, stroke only)
ICONS = {
    "chat": '<path d="M4 5h16v10H9l-5 4z"/><path d="M8 9h8M8 12h5"/>',
    "cite": '<path d="M10 14a4 4 0 0 1 0-5.6l2.4-2.4a4 4 0 0 1 5.6 5.6l-1.2 1.2"/>'
            '<path d="M14 10a4 4 0 0 1 0 5.6l-2.4 2.4a4 4 0 0 1-5.6-5.6l1.2-1.2"/>',
    "compare": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    "shield": '<path d="M12 3l8 3v6c0 4.5-3.4 8-8 9-4.6-1-8-4.5-8-9V6z"/><path d="M9 12l2 2 4-4"/>',
}


def icon(name: str) -> str:
    return (f'<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    :root {
        --ink: #0b1b3a;
        --muted: #5b6b85;
        --line: #e3e8f0;
        --soft: #f5f8fc;
        --accent: #10b981;
        --accent-dark: #059669;
        --accent-soft: #ecfdf5;
    }
    html, body, .stApp, .stApp p, .stApp button, .stApp input {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: #ffffff;
        color: var(--ink);
    }
    header[data-testid="stHeader"] {
        display: none;
    }
    .block-container {
        padding: 0 0 2rem 0;
        max-width: 100%;
    }
    /* The zero-height component that runs the hero animation */
    div[data-testid="stElementContainer"]:has(iframe[srcdoc*="hero-net"]) {
        position: absolute;
        height: 0;
        overflow: hidden;
    }
    .nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.9rem 3rem;
        background: #ffffff;
        border-bottom: 1px solid var(--line);
    }
    .brand, .brand:hover {
        text-decoration: none;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-weight: 800;
        font-size: 1rem;
        letter-spacing: 0.14em;
        color: var(--ink);
    }
    .nav-links a {
        color: var(--muted);
        text-decoration: none;
        font-size: 0.92rem;
        font-weight: 500;
        margin-left: 1.5rem;
    }
    .nav-links a:hover {
        color: var(--accent-dark);
    }
    .nav-links {
        display: flex;
        align-items: center;
    }
    .nav-links a.nav-icon {
        display: inline-flex;
        color: var(--ink);
    }
    .nav-links a.nav-icon:hover {
        color: var(--accent-dark);
    }
    .nav-links a.nav-button {
        color: #ffffff;
        background: var(--ink);
        padding: 0.5rem 1rem;
        border-radius: 8px;
    }
    .nav-links a.nav-button:hover {
        background: var(--accent-dark);
        color: #ffffff;
    }
    .hero {
        position: relative;
        overflow: hidden;
        background: radial-gradient(circle at 20% 20%, #16306b 0%, rgba(22, 48, 107, 0) 55%),
                    radial-gradient(circle at 85% 80%, #0d4f4a 0%, rgba(13, 79, 74, 0) 50%),
                    linear-gradient(160deg, #071226 0%, #0b1b3a 60%, #0a2a33 100%);
        color: #ffffff;
        text-align: center;
        padding: 110px 2rem 96px 2rem;
    }
    .hero > *:not(canvas) {
        position: relative;
        z-index: 1;
    }
    .hero.compact {
        padding: 40px 2rem 36px 2rem;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(52, 211, 153, 0.45);
        color: #a7f3d0;
        padding: 7px 18px;
        border-radius: 50px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        margin-bottom: 1.75rem;
    }
    .hero-badge .dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #34d399;
        box-shadow: 0 0 10px #34d399;
    }
    .hero.compact .hero-badge {
        margin-bottom: 0.9rem;
    }
    .hero h1 {
        color: #ffffff;
        font-size: 3.4rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.12;
        margin: 0 auto 1.25rem auto;
        max-width: 860px;
        padding: 0;
    }
    .hero h1 span {
        background: linear-gradient(90deg, #34d399, #38bdf8);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .hero.compact h1 {
        font-size: 1.9rem;
        margin-bottom: 0.4rem;
    }
    .hero p {
        font-size: 1.2rem;
        color: #c7d2e5;
        max-width: 700px;
        margin: 0 auto 2.25rem auto;
        line-height: 1.6;
    }
    .hero.compact p {
        font-size: 0.95rem;
        margin-bottom: 0;
    }
    .hero-btn {
        display: inline-block;
        background: var(--accent);
        color: #04241a !important;
        padding: 14px 34px;
        border-radius: 12px;
        text-decoration: none;
        font-weight: 700;
        font-size: 1.05rem;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.35);
        transition: all 0.25s ease;
    }
    .hero-btn:hover {
        transform: translateY(-2px);
        background: #34d399;
        box-shadow: 0 14px 36px rgba(16, 185, 129, 0.5);
    }
    .hero-meta {
        margin-top: 1.6rem;
        font-size: 0.85rem;
        color: #93a4c0;
    }
    .section {
        max-width: 1100px;
        margin: 0 auto;
        padding: 72px 2rem 0 2rem;
    }
    .section .kicker {
        text-align: center;
        color: var(--accent-dark);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .section h2 {
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: var(--ink);
        margin: 0 0 0.5rem 0;
        padding: 0;
    }
    .section .lead {
        text-align: center;
        color: var(--muted);
        font-size: 1.05rem;
        max-width: 680px;
        margin: 0 auto 2.25rem auto;
    }
    .features {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1rem;
    }
    .feature {
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1.4rem 1.25rem;
        background: #ffffff;
        transition: all 0.2s ease;
    }
    .feature:hover {
        border-color: var(--accent);
        box-shadow: 0 10px 28px rgba(11, 27, 58, 0.08);
        transform: translateY(-2px);
    }
    .feature .icon {
        display: inline-flex;
        padding: 0.55rem;
        border-radius: 10px;
        background: var(--accent-soft);
        color: var(--accent-dark);
        margin-bottom: 0.8rem;
    }
    .feature h4, .step h4 {
        margin: 0 0 0.4rem 0;
        padding: 0;
        font-size: 1rem;
        font-weight: 700;
        color: var(--ink);
    }
    .feature p, .step p {
        margin: 0;
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.55;
    }
    .steps {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
    }
    .step {
        background: var(--soft);
        border-radius: 14px;
        padding: 1.4rem 1.25rem;
    }
    .step .num {
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        color: var(--accent-dark);
        margin-bottom: 0.6rem;
    }
    .sample {
        max-width: 820px;
        margin: 0 auto;
        border: 1px solid var(--line);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(11, 27, 58, 0.07);
    }
    .sample-q {
        background: var(--soft);
        padding: 1rem 1.4rem;
        font-weight: 700;
        color: var(--ink);
        border-bottom: 1px solid var(--line);
    }
    .sample-a {
        padding: 1.2rem 1.4rem;
        line-height: 1.7;
        color: var(--ink);
    }
    .sample-sources {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem 1.5rem;
        padding: 0.9rem 1.4rem 1.2rem 1.4rem;
        font-size: 0.85rem;
        color: var(--muted);
    }
    .cite {
        display: inline-block;
        min-width: 1.3rem;
        padding: 0 0.3rem;
        border-radius: 6px;
        background: var(--accent-soft);
        color: var(--accent-dark);
        font-size: 0.78rem;
        font-weight: 700;
        text-align: center;
        margin-right: 0.2rem;
    }
    .cache-badge {
        margin-left: 0.6rem;
        padding: 0.1rem 0.55rem;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent-dark);
        letter-spacing: 0.02em;
        text-transform: none;
    }
    .qa {
        display: inline-block;
        width: 1.6rem;
        height: 1.6rem;
        line-height: 1.6rem;
        text-align: center;
        border-radius: 6px;
        background: var(--ink);
        color: #ffffff;
        font-size: 0.8rem;
        font-weight: 800;
        margin-right: 0.6rem;
    }
    .sample-a .qa {
        background: var(--accent);
        color: #04241a;
    }
    /* Streamlit adds a link icon next to every heading; not wanted on this page */
    [data-testid="stHeaderActionElements"] {
        display: none !important;
    }
    .cta {
        text-align: center;
        padding-top: 60px;
    }
    @media (max-width: 900px) {
        .features { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .steps { grid-template-columns: 1fr; }
        .hero h1 { font-size: 2.2rem; }
        .nav { padding: 0.9rem 1rem; }
        .nav-links a { margin-left: 1rem; }
    }
    @media (max-width: 560px) {
        .features { grid-template-columns: 1fr; }
    }
    .section-label {
        color: var(--muted);
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin: 1.5rem 0 0.5rem 0;
    }
    .question {
        font-size: 1.4rem;
        font-weight: 700;
        line-height: 1.35;
        margin: 1.75rem 0 0.25rem 0;
        color: var(--ink);
    }
    .footer {
        max-width: 1100px;
        margin: 4rem auto 0 auto;
        padding: 1.25rem 2rem 0.5rem 2rem;
        border-top: 1px solid var(--line);
        color: var(--muted);
        font-size: 0.85rem;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        gap: 0.5rem;
    }
    .footer a {
        color: var(--muted);
        text-decoration: none;
        margin-left: 1rem;
    }
    .footer a:hover {
        color: var(--accent-dark);
    }
    div[data-testid="stButton"] button {
        border: 1px solid var(--line);
        background: #ffffff;
        color: var(--ink);
        border-radius: 12px;
        min-height: 4.2rem;
        transition: all 0.15s ease-in-out;
    }
    div[data-testid="stButton"] button:hover {
        border-color: var(--accent);
        color: var(--accent-dark);
        background: var(--accent-soft);
    }
    div[data-testid="stButton"] button p {
        font-size: 0.86rem;
        text-align: left;
    }
    div[data-testid="stFormSubmitButton"] button {
        background: var(--accent);
        border: none;
        color: #04241a;
        border-radius: 10px;
        font-weight: 700;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        background: #34d399;
        color: #04241a;
    }
    div[data-testid="stForm"] {
        border: none;
        padding: 0;
    }
    div[data-testid="stTextInput"] input {
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# A network of drifting points behind the hero text; points move away from the mouse and the ones
# near it are linked to it. Streamlit markdown cannot run scripts, so this runs in a zero-height
# component and draws on a canvas it adds to the hero in the main page (the component iframe has
# the same origin). If that ever fails, the hero keeps its plain gradient background.
HERO_ANIMATION = """
<script>
(function () {
  const win = window.parent, doc = win.document;
  function start(hero) {
    const canvas = doc.createElement("canvas");
    canvas.className = "hero-net";
    canvas.style.cssText = "position:absolute;inset:0;width:100%;height:100%;z-index:0;pointer-events:none;";
    hero.prepend(canvas);
    const ctx = canvas.getContext("2d");
    const mouse = {x: null, y: null};
    let points = [], w = 0, h = 0;
    function resize() {
      const ratio = win.devicePixelRatio || 1;
      w = hero.clientWidth; h = hero.clientHeight;
      canvas.width = w * ratio; canvas.height = h * ratio;
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      const count = Math.min(120, Math.round(w * h / 9000));
      points = Array.from({length: count}, () => ({
        x: Math.random() * w, y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.35, vy: (Math.random() - 0.5) * 0.35}));
    }
    hero.addEventListener("mousemove", (e) => {
      const r = hero.getBoundingClientRect();
      mouse.x = e.clientX - r.left; mouse.y = e.clientY - r.top;
    });
    hero.addEventListener("mouseleave", () => { mouse.x = null; mouse.y = null; });
    resize();
    let lastW = w;
    function frame() {
      if (!canvas.isConnected) return;  // Streamlit redrew the hero; a new canvas takes over
      if (hero.clientWidth !== lastW) { resize(); lastW = w; }
      ctx.clearRect(0, 0, w, h);
      for (const p of points) {
        if (mouse.x !== null) {
          const dx = p.x - mouse.x, dy = p.y - mouse.y, d = Math.hypot(dx, dy);
          if (d < 120 && d > 0) { p.x += dx / d * 1.2; p.y += dy / d * 1.2; }
        }
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0 || p.x > w) p.vx *= -1;
        if (p.y < 0 || p.y > h) p.vy *= -1;
      }
      for (let i = 0; i < points.length; i++) {
        const a = points[i];
        for (let j = i + 1; j < points.length; j++) {
          const b = points[j], d = Math.hypot(a.x - b.x, a.y - b.y);
          if (d < 130) {
            ctx.strokeStyle = "rgba(125, 211, 252," + (0.22 * (1 - d / 130)) + ")";
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
          }
        }
        if (mouse.x !== null) {
          const d = Math.hypot(a.x - mouse.x, a.y - mouse.y);
          if (d < 190) {
            ctx.strokeStyle = "rgba(52, 211, 153," + (0.55 * (1 - d / 190)) + ")";
            ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(mouse.x, mouse.y); ctx.stroke();
          }
        }
        ctx.fillStyle = "rgba(167, 243, 208, 0.75)";
        ctx.beginPath(); ctx.arc(a.x, a.y, 1.6, 0, Math.PI * 2); ctx.fill();
      }
      win.requestAnimationFrame(frame);
    }
    win.requestAnimationFrame(frame);
  }
  function attach() {
    doc.querySelectorAll(".hero").forEach((hero) => {
      if (!hero.querySelector("canvas.hero-net")) start(hero);
    });
  }
  attach();
  win.setInterval(attach, 700);
})();
</script>
"""

url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/query")
backend_api_key = os.getenv("BACKEND_API_KEY", "")

COMPANY_NAMES = {"AAPL": "Apple", "MSFT": "Microsoft"}
INDEXED_YEARS = ["2024", "2025"]
EXAMPLE_QUERIES = [
    "Compare Apple's and Microsoft's total net revenue for fiscal year 2024.",
    "How did Apple's net income change between fiscal 2024 and fiscal 2025?",
    "What were Microsoft's primary cybersecurity risks in 2024?",
]
FEATURES = [
    ("chat", "Plain-English questions", "Ask about revenue, profits, risk factors or strategy the way you would "
                                       "ask an analyst. No query syntax."),
    ("cite", "Cited answers", "Every figure is cited to the 10-K section it comes from, with a link to the "
                             "filing on sec.gov."),
    ("compare", "Comparisons", "Compare companies and fiscal years in one question; each company and year is "
                              "searched separately so none is left out."),
    ("shield", "Guardrails", "Questions outside the indexed filings and prompt injection attempts get a clear "
                            "refusal instead of a made-up answer."),
]
STEPS = [
    ("Route", "A small model turns the question into filters: which companies, fiscal years and 10-K "
              "sections to search."),
    ("Retrieve", "The matching filing excerpts are found by meaning (vector search) within those filters."),
    ("Answer", "A larger model writes the answer from the excerpts only, citing each figure to its source."),
]
# A real answer from the evaluation run (golden set item cmp-01), shown on the landing page
SAMPLE_ANSWER = (
    "<div class=\"sample\"><div class=\"sample-q\"><span class=\"qa\">Q</span>“Compare Apple's and Microsoft's "
    "total net revenue for fiscal year 2024.”</div><div class=\"sample-a\"><span class=\"qa\">A</span>Apple's total net sales for fiscal 2024 were <b>$391,035 million</b> "
    "<span class=\"cite\">1</span>. Microsoft's total revenue for fiscal 2024 was <b>$245,122 million</b> "
    "<span class=\"cite\">2</span>. Difference: $391,035 m − $245,122 m = <b>$145,913 million</b>; Apple's "
    "revenue is about <b>59.5% higher</b> (391,035 ÷ 245,122 ≈ 1.595).</div>"
    '<div class="sample-sources"><span><span class="cite">1</span> AAPL FY2024 · Item 8 · Consolidated '
    'Statements of Operations</span><span><span class="cite">2</span> MSFT FY2024 · Item 8 · Income '
    'Statements</span></div></div>'
)
ERROR_MESSAGES = {
    429: "Too many requests. Please wait a minute and try again.",
    503: "The language model service is unavailable right now. Please try again shortly.",
    422: "The question must be between 1 and 500 characters.",
}
# Inline citation written by the synthesizer, e.g. [AAPL | FY2024 | Item 8]
CITATION = re.compile(r"\[([A-Z]{1,5}) \| FY(\d{4}) \| (Item \d{1,2}[A-C]?|General)\]")


# The backend only sees this container's IP, so pass on the end user's IP for its per-user rate
# limit. Azure's ingress appends the client IP as the last X-Forwarded-For entry; earlier entries
# can be set by the client itself, so only the last one is used.
def end_user_ip() -> str | None:
    forwarded_for = st.context.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",")[-1].strip()
    return st.context.ip_address


def ask_backend(question: str) -> tuple[dict | None, str | None]:
    # Returns (result, None) or (None, an error message for the user)
    headers = {"X-Internal-Key": backend_api_key}
    client_ip = end_user_ip()
    if client_ip:
        headers["X-End-User-IP"] = client_ip
    try:
        response = requests.post(url, json={"query": question}, headers=headers, timeout=180)
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        logger.warning("Backend returned HTTP %s", response.status_code)
        return None, ERROR_MESSAGES.get(response.status_code,
                                        "Something went wrong while generating the answer. Please try again.")
    except requests.exceptions.RequestException:
        # Details (including the backend URL) go to the server log, not to the user
        logger.exception("Could not reach the backend")
        return None, "The service is not reachable right now. Please try again in a minute."
    return response.json(), None



def number_citations(answer: str, sources: list) -> str:
    # Replaces each [AAPL | FY2024 | Item 8] label with the number of its source card. Labels the
    # backend did not turn into a source (e.g. made up by the model) stay as they are.
    numbers = {(s["company"], s["year"], s["section"]): i for i, s in enumerate(sources, start=1)}

    def replace(match: re.Match) -> str:
        n = numbers.get((match.group(1), match.group(2), match.group(3)))
        return f":green-background[{n}]" if n else match.group(0)

    text = CITATION.sub(replace, answer)
    # The same source cited twice in a row is shown once
    return re.sub(r"(:green-background\[\d+\])(\s*\1)+", r"\1", text)


def fill_question(question: str):
    # Example button callback: puts the example in the text box (it runs before the box is drawn
    # again); the visitor still presses Ask
    st.session_state.query_input = question


for key, default in [("query_input", ""), ("pending", None), ("asked", None), ("result", None), ("error", None)]:
    st.session_state.setdefault(key, default)

# "Get started" links to ?start=1 on the same page, which swaps the landing sections for the
# question screen
started = st.query_params.get("start") == "1"

st.markdown(
    f'<div class="nav"><a class="brand" href="?" target="_self">{LOGO_SVG}FINANCIAL RAG</a><div class="nav-links">'
    f'<a class="nav-icon" href="{GITHUB_PROFILE_URL}" target="_blank" title="GitHub profile">{GITHUB_ICON}</a>'
    f'<a class="nav-icon" href="{LINKEDIN_URL}" target="_blank" title="LinkedIn">{LINKEDIN_ICON}</a>'
    f'<a class="nav-button" href="{REPO_URL}" target="_blank">View the code</a></div></div>',
    unsafe_allow_html=True,
)

companies = " · ".join(f"{name} ({ticker})" for ticker, name in COMPANY_NAMES.items())
coverage = f"Covering {companies} · fiscal years {INDEXED_YEARS[0]}–{INDEXED_YEARS[-1]}"
badge = '<div class="hero-badge"><span class="dot"></span>AI-POWERED RESEARCH</div>'
if started:
    st.markdown(
        f'<div class="hero compact">{badge}<h1>Ask the <span>10-K</span></h1><p>{html.escape(coverage)}</p></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="hero">{badge}<h1>Ask the <span>10-K</span>.<br>Get answers you can verify.</h1>'
        '<p>Plain-English questions over annual reports filed with the SEC, answered with every figure '
        'cited to the filing it comes from.</p>'
        '<a class="hero-btn" href="?start=1" target="_self">Get started →</a>'
        f'<div class="hero-meta">{html.escape(coverage)}</div></div>',
        unsafe_allow_html=True,
    )
st.iframe(HERO_ANIMATION, height=1)

if not started:
    cards = "".join(f'<div class="feature"><div class="icon">{icon(name)}</div><h4>{html.escape(title)}</h4>'
                    f'<p>{html.escape(text)}</p></div>' for name, title, text in FEATURES)
    steps = "".join(f'<div class="step"><div class="num">STEP {i}</div><h4>{html.escape(title)}</h4>'
                    f'<p>{html.escape(text)}</p></div>' for i, (title, text) in enumerate(STEPS, start=1))
    st.markdown(
        '<div class="section"><div class="kicker">What it does</div><h2>Research filings in seconds</h2>'
        '<p class="lead">Instead of searching a 100-page annual report, ask a question and get the figures '
        'with their sources.</p>'
        f'<div class="features">{cards}</div></div>'
        '<div class="section"><div class="kicker">Under the hood</div><h2>How it works</h2>'
        '<p class="lead">A retrieval-augmented generation pipeline over the SEC filings.</p>'
        f'<div class="steps">{steps}</div></div>'
        f'<div class="section"><div class="kicker">See it in action</div><h2>An example answer</h2>'
        '<p class="lead">A real answer from the system, with each figure cited to its filing.</p>'
        f'{SAMPLE_ANSWER}</div>'
        '<div class="section cta"><a class="hero-btn" href="?start=1" target="_self">Ask your first question →</a></div>',
        unsafe_allow_html=True,
    )
else:
    _, center, _ = st.columns([1, 5, 1])
    with center:
        st.markdown('<div style="height: 1.75rem"></div>', unsafe_allow_html=True)
        # Question box: Enter or the button submits
        with st.form("ask_form", clear_on_submit=False):
            input_col, button_col = st.columns([6, 1], vertical_alignment="bottom")
            input_col.text_input("Question", key="query_input", max_chars=500, label_visibility="collapsed",
                                 placeholder="Ask a question about the filings…")
            if button_col.form_submit_button("Ask", use_container_width=True):
                st.session_state.pending = st.session_state.query_input

        # The examples always stay in the same place, above the answer
        st.markdown('<div class="section-label">Try an example (fills in the box)</div>', unsafe_allow_html=True)
        example_cols = st.columns(len(EXAMPLE_QUERIES))
        for i, (col, example) in enumerate(zip(example_cols, EXAMPLE_QUERIES)):
            col.button(example, key=f"example_{i}", on_click=fill_question, args=(example,), use_container_width=True)

        pending = st.session_state.pending
        if pending is not None:
            st.session_state.pending = None
            if not pending.strip():
                st.warning("Please enter a question first.")
            else:
                with st.spinner("Searching the filings and writing the answer… The first question after a quiet "
                                "period can take a minute or two while the backend wakes up."):
                    result, error = ask_backend(pending)
                st.session_state.asked, st.session_state.result, st.session_state.error = pending, result, error

        if st.session_state.error:
            st.error(st.session_state.error)
        elif st.session_state.result:
            result = st.session_state.result
            sources = result.get("sources") or []
            # The question is HTML-escaped and the answer rendered as Markdown, never as raw HTML, so any
            # HTML in them (e.g. a prompt-injected <script>) is displayed, not executed. A bare "$" starts
            # a LaTeX math span in Streamlit, so it is escaped.
            st.markdown(f'<div class="question">{html.escape(st.session_state.asked)}</div>', unsafe_allow_html=True)

            cache_badge = ('<span class="cache-badge" title="This question was asked before, so no model call '
                           'was needed">⚡ from cache</span>') if result.get("cached") else ""
            st.markdown(f'<div class="section-label">Answer{cache_badge}</div>', unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(number_citations(result["answer"].replace("$", "\\$"), sources))

            if sources:
                st.markdown('<div class="section-label">Sources</div>', unsafe_allow_html=True)
                for i, source in enumerate(sources, start=1):
                    with st.container(border=True):
                        title = f":green-background[{i}] **{source['company']} FY{source['year']}** · {source['section']}"
                        link = source.get("passage_url") or source.get("url") or ""
                        if link.startswith("https://www.sec.gov/"):
                            title += f" · [Open in the 10-K ↗]({link})"
                        st.markdown(title)
                        st.caption(source["snippet"].replace("$", "\\$") + "…")

st.markdown(
    f'<div class="footer"><div>Built by Ayhan Meherrem · Based on public SEC filings. Not investment advice.<br>'
    f'The backend sleeps when idle, so the first question after a quiet period is slower.</div>'
    f'<div><a href="{REPO_URL}" target="_blank">Source code</a><a href="{GITHUB_PROFILE_URL}" target="_blank">GitHub</a>'
    f'<a href="{LINKEDIN_URL}" target="_blank">LinkedIn</a></div></div>',
    unsafe_allow_html=True,
)
