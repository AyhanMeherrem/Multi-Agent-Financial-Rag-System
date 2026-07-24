import os
import sys
sys.path.append(".")
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import VectorStoreIndex
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
    return Groq(model="llama-3.1-8b-instant", api_key=api_key)

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


""""""""""""""""""""""""""""""""""""""""HELPER TEST FUNCTION """""""""""""""""

from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def load_index_from_qdrant() -> VectorStoreIndex:
    client = QdrantClient(path="./data/qdrant_db")
    vector_store = QdrantVectorStore(client=client, collection_name="financial_filings")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
    return VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)

if __name__ == "__main__":
    index = load_index_from_qdrant()
    sample_query = "What were Apple's primary risk factors in 2024?"
    nodes, filters = route_query_with_llm(sample_query, index)
    
    print("\n--- Router Agent Execution Results ---")
    print(f"Extracted Metadata Filters: {filters}")
    print(f"Total Filtered Nodes Retrieved: {len(nodes)}")
    if nodes:
        print(f"Top Retrieved Node Snippet: {nodes[0].node.text[:200]}...")


