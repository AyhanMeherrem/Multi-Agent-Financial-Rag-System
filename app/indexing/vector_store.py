import sys
sys.path.append(".")
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from app.parsing.parse_filings import parse_all_filings

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

client = QdrantClient(path="./data/qdrant_db")
vector_store = QdrantVectorStore(client=client, collection_name="financial_filings")

storage_context = StorageContext.from_defaults(vector_store=vector_store)   # Basically pass which store the data will be stored

nodes = parse_all_filings()
index = VectorStoreIndex(
    nodes=nodes,
    embed_model=embed_model,
    storage_context=storage_context
)

print("Index created successfully")