import os
import sys
sys.path.append(".")
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import VectorStoreIndex
from vector_store import storage_context

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