from contextlib import asynccontextmanager  # For lifespan event managing
from fastapi import FastAPI
from fastapi.requests import Request
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.router.router_agent import load_index_from_qdrant
from app.synthesizer.synthesizer_agent import synthesize_financial_answer

class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)

class AgentResponse(BaseModel):
    answer : str
    companies: list[str] | None =None
    year : str | None = None

@asynccontextmanager
async def lifespan(app:FastAPI):
    app.state.index = load_index_from_qdrant() # runs only one time before backend starting
    yield
    # Do nothing special for shut down

app = FastAPI(title='Financial RAG System', lifespan=lifespan)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/query", response_model=AgentResponse)
@limiter.limit("10/minute")
async def financial_query(request: Request, body: QueryRequest):
    index = request.app.state.index
    final_answer, filters = synthesize_financial_answer(body.query, index)
    return AgentResponse(answer=final_answer, companies=filters.get("companies"), year=filters.get("year"))
