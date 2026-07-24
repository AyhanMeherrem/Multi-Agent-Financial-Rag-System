import sys
sys.path.append(".")
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from app.parsing.parse_filings import parse_all_filings
from llama_index.core.node_parser import SentenceSplitter   # To apply subchunking if we have a paragraph which has >512 tokens which is limit of structural chunking

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

client = QdrantClient(path="./data/qdrant_db")
vector_store = QdrantVectorStore(client=client, collection_name="financial_filings")

storage_context = StorageContext.from_defaults(vector_store=vector_store)   # Basically pass which store the data will be stored

raw_nodes = parse_all_filings()
print(f"Extracted {len(raw_nodes)} raw section nodes.")

# Sub Chunking if it has more tahn 512 token
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)  # 50 token overlap makes easier to catch meanings 
nodes = splitter.get_nodes_from_documents(raw_nodes)
print(f"Refined into {len(nodes)} 512 token chunks")



index = VectorStoreIndex(
    nodes=nodes,
    embed_model=embed_model,
    storage_context=storage_context,
    show_progress=True  #just for visualizing the progress
)

print("Index created successfully and its in ./data/qdrant_db")