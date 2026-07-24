import os
import sys
sys.path.append(".")
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import VectorStoreIndex
from vector_store import storage_context
import json # For parsing filtering metadata from user query  text


load_dotenv()

def get_router_llm() -> Groq:
    """
    Initializes Groq LPU LLM Engine (llama-3.1-8b) for metadata parsing and routing.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY is missing or unconfigured in .env file!"
        )
    return Groq(model="llama-3.1-8b", api_key=api_key)

def construct_sec_filters(company:str=None, year:str=None, section:str=None) -> MetadataFilters:
    filters = []
    if company:
        filters.append(ExactMatchFilter(key="company", value=company))
    if year:
        filters.append(ExactMatchFilter(key="year", value=str(year)))
    if section:
        filters.append(ExactMatchFilter(key="section", value=section))
    return MetadataFilters(filters=filters, condition='and')
    # using none bc if user asks without specified year or company we don't wanna filter them

def get_sec_retriever(index: VectorStoreIndex, company:str=None, year:str=None, section:str=None, top_k:int=5) -> VectorIndexRetriever:
    metadata_filters = construct_sec_filters(company, year, section)
    return VectorIndexRetriever(index=index, similarity_top_k=top_k, filters=metadata_filters)

def route_query_with_llm(query_string: str, index: VectorStoreIndex):
    llm = get_router_llm()

    prompt = f"""You are an SEC 10-K query router agent.
        Extract metadata entities from the user's question:
        - "company": "AAPL", "MSFT", or null.
        - "year": "2024", "2025", or null.
        - "section": "Item 1A", "Item 7", "Item 8", or null.
        Respond ONLY with a valid raw JSON object.
        Example: {{"company": "AAPL", "year": "2024", "section": "Item 1A"}}
        Question: "{query_string}"
        JSON Output:"""

    response = llm.complete(prompt)
    clean_json = response.text.strip().replace("```json", "").replace("```", "").strip()
    
    filter_dict = json.loads(clean_json)

    retriever = get_sec_retriever(index, 
    company=filter_dict.get("company"), 
    year=str(filter_dict.get("year")) if filter_dict.get("year") else None, 
    section=filter_dict.get("section"), 
    top_k=5)

    nodes = retriever.retrieve(query_string)
    return nodes, filter_dict