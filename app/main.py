from contextlib import asynccontextmanager  # For lifespan event managing
from fastapi import FastAPI
from fastapi.requests import Request
import json
from pydantic import BaseModel
from app.router.router_agent import load_index_from_qdrant
from app.synthesizer.synthesizer_agent import synthesize_financial_answer

class QueryRequest(BaseModel):
    query:str

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

@app.post("/query",response_model= AgentResponse)
async def financial_query(body:QueryRequest, request: Request):
    index = request.app.state.index
    final_answer, filters = synthesize_financial_answer(body.query, index)
    return AgentResponse(answer=final_answer, companies=filters["companies"], year=filters["year"])

    