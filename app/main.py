import ipaddress
import logging
import os
import secrets
from contextlib import asynccontextmanager  # For lifespan event managing
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.requests import Request
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.answer_cache import AnswerCache
from app.llm_errors import LLMRateLimited, LLMUnavailable
from app.router.router_agent import get_catalog, load_index_from_qdrant
from app.synthesizer.synthesizer_agent import answer_query, build_sources

logger = logging.getLogger("uvicorn.error")


TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "false").lower() == "true"
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY")

# The backend only ever talks to the Streamlit frontend, so the connection IP is the frontend's
# and every end user would share one rate-limit bucket. The frontend therefore sends the end
# user's IP in this header.
END_USER_IP_HEADER = "x-end-user-ip"


def has_valid_internal_key(request: Request) -> bool:
    key = request.headers.get("x-internal-key")
    return bool(BACKEND_API_KEY and key and secrets.compare_digest(key, BACKEND_API_KEY))


# Rate-limit key. The forwarded end-user IP is only trusted when TRUST_PROXY_HEADERS is on AND the
# request carries the internal key, i.e. it really came from our frontend. Anyone else could set
# the header to a new value per request and bypass the limit, so they get their connection IP.
def get_client_ip(request: Request) -> str:
    if TRUST_PROXY_HEADERS and has_valid_internal_key(request):
        forwarded = request.headers.get(END_USER_IP_HEADER, "").strip()
        try:
            return str(ipaddress.ip_address(forwarded))
        except ValueError:
            pass
    return request.client.host if request.client else "unknown"


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)

class Source(BaseModel):
    company: str
    year: str
    section: str
    url: str | None = None  # the filing document on sec.gov
    snippet: str

class AgentResponse(BaseModel):
    answer : str
    companies: list[str] | None =None
    year : str | None = None  # set when the question is about exactly one fiscal year
    years: list[str] | None = None
    source_urls: dict[str, str] | None = None  # EDGAR company search links, kept for compatibility
    sources: list[Source] | None = None  # the filing excerpts the answer cites
    cached: bool = False  # served from the answer cache, no LLM call

# just building visualization for pdfs, it returns url's under answer box
def build_edgar_source_urls(companies: list[str] | None, year: str | None) -> dict[str, str] | None:
    # Links for each company's 10-K pdfs on SEC EDGAR website, not the specific chunk which needs accession-number metadata threaded through parsingindexing.
    # Might be improved in future to return specific chunks but so far enough
    if not companies:
        return None
    urls = {}
    for company in companies:
        params = f"action=getcompany&CIK={company}&type=10-K"
        if year and year.isdigit():
            params += f"&datea={year}0101&dateb={int(year) + 1}1231"
        urls[company] = f"https://www.sec.gov/cgi-bin/browse-edgar?{params}"
    return urls

@asynccontextmanager
async def lifespan(app:FastAPI):
    app.state.index = load_index_from_qdrant() # runs only one time before backend starting
    get_catalog(app.state.index)  # read the indexed companies/years/sections once, up front
    app.state.answer_cache = AnswerCache()
    yield
    # Do nothing special for shut down

app = FastAPI(title='Financial RAG System', lifespan=lifespan)

# Limiter state lives in this process's memory: it resets on restart and is not shared between
# replicas. Fine for a single replica; multiple replicas would need a shared store.
limiter = Limiter(key_func=get_client_ip)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def verify_internal_key(x_internal_key: str | None = Header(None)):
    # Only the frontend container (which reads the same env var) knows this value 
    if not BACKEND_API_KEY:
        # an unconfigured key must never mean "no auth required"
        raise HTTPException(status_code=503, detail="Backend not configured.")
    if not x_internal_key or not secrets.compare_digest(x_internal_key, BACKEND_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing internal key.")


# No auth and no rate limit: used by container health probes and for checking the app is up
@app.get("/health")
async def health(request: Request):
    return {"status": "ok", "index_loaded": getattr(request.app.state, "index", None) is not None}


@app.post("/query", response_model=AgentResponse, dependencies=[Depends(verify_internal_key)])
@limiter.limit("10/minute")
async def financial_query(request: Request, body: QueryRequest):
    index = request.app.state.index
    cache = request.app.state.answer_cache
    cached = cache.get(body.query)
    if cached is not None:
        return cached.model_copy(update={"cached": True})
    try:
        # The pipeline is blocking (Groq HTTP calls, CPU embedding, local Qdrant). Running it in
        # a worker thread keeps the event loop free for other requests such as /health.
        final_answer, filters, nodes = await run_in_threadpool(answer_query, body.query, index)
    except LLMRateLimited:
        logger.warning("Groq rate limit reached")
        raise HTTPException(status_code=429, detail="The language model is busy. Please try again in a minute.")
    except LLMUnavailable:
        logger.exception("Groq unavailable")
        raise HTTPException(status_code=503, detail="The language model service is unavailable. Please try again shortly.")
    except Exception:
        # catch any error during synthesis and return 502 Bad Gateway 
        logger.exception("financial_query failed")
        raise HTTPException(status_code=502, detail="Failed to generate an answer. Please try again.")
    companies = filters.get("companies")
    year = filters.get("year")
    # Only link filings that are actually indexed (the router also reports non-indexed tickers), and
    # none at all when the answer is a fixed "not covered" or "out of scope" message
    indexed = [] if filters.get("fixed_answer") else [c for c in companies or [] if c in get_catalog(index).companies]
    response = AgentResponse(
        answer=final_answer,
        companies=companies,
        year=year,
        years=filters.get("years"),
        source_urls=build_edgar_source_urls(indexed, year),
        sources=build_sources(final_answer, nodes),
    )
    # Errors raise above, so only real answers are cached
    cache.put(body.query, response)
    return response
