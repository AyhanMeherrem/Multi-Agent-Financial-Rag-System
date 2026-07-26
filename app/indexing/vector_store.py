import os
import shutil
import sys
sys.path.append(".")
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from app.parsing.parse_filings import parse_all_filings
from llama_index.core.node_parser import SentenceSplitter   # To apply subchunking if we have a paragraph which has >512 tokens which is limit of structural chunking

QDRANT_DB_PATH = "./data/qdrant_db"


def build_index() -> VectorStoreIndex:
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
    # delete_collection() + reuse of the same collection name (verified: old points
    # resurface and merge with newly inserted ones). Wiping the on-disk directory
    # before opening a client is the only way to guarantee a truly clean rebuild.
    if os.path.exists(QDRANT_DB_PATH):
        print(f"Cleaning duped points from {QDRANT_DB_PATH}  ")
        shutil.rmtree(QDRANT_DB_PATH)

    client = QdrantClient(path=QDRANT_DB_PATH)

    try:
        vector_store = QdrantVectorStore(client=client, collection_name="financial_filings")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

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
            show_progress=True  # just for visualizing the progress
        )

        print("Index created successfully and its in ./data/qdrant_db")
        return index
    finally:
        #  Qdrant must release its file lock cleanly, or the next
        # process to open this path can silently fail. see existing data and append
        # instead of replacing it (this caused the point count duplication bug)
        client.close()


if __name__ == "__main__":
    build_index()
