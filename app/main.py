import logging
import os
import secrets
from contextlib import asynccontextmanager  # For lifespan event managing
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.requests import Request
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.router.router_agent import load_index_from_qdrant
from app.synthesizer.synthesizer_agent import synthesize_financial_answer

logger = logging.getLogger("uvicorn.error")


TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "false").lower() == "true"

# gets ip of user requesting (frontend or anything else) for rate limiting
def get_client_ip(request: Request) -> str:
    if TRUST_PROXY_HEADERS:
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)

class AgentResponse(BaseModel):
    answer : str
    companies: list[str] | None =None
    year : str | None = None
    source_urls: dict[str, str] | None = None

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
    yield
    # Do nothing special for shut down

app = FastAPI(title='Financial RAG System', lifespan=lifespan)

limiter = Limiter(key_func=get_client_ip)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

BACKEND_API_KEY = os.getenv("BACKEND_API_KEY")


def verify_internal_key(x_internal_key: str | None = Header(None)):
    # Only the frontend container (which reads the same env var) knows this value 
    if not BACKEND_API_KEY:
        # an unconfigured key must never mean "no auth required"
        raise HTTPException(status_code=503, detail="Backend not configured.")
    if not x_internal_key or not secrets.compare_digest(x_internal_key, BACKEND_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing internal key.")


@app.post("/query", response_model=AgentResponse, dependencies=[Depends(verify_internal_key)])
@limiter.limit("10/minute")
async def financial_query(request: Request, body: QueryRequest):
    index = request.app.state.index
    try:
        final_answer, filters = synthesize_financial_answer(body.query, index)
    except Exception:
        # catch any error during synthesis and return 502 Bad Gateway 
        logger.exception("financial_query failed")
        raise HTTPException(status_code=502, detail="Failed to generate an answer. Please try again.")
    companies = filters.get("companies")
    year = filters.get("year")
    return AgentResponse(
        answer=final_answer,
        companies=companies,
        year=year,
        source_urls=build_edgar_source_urls(companies, year),
    )
