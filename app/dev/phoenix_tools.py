# Dev-only tooling: Arize Phoenix tracing, a vector dataset upload for Phoenix's 3D view, and an
# interactive terminal session. Nothing in the serving path imports this module, so the backend
# image does not need arize-phoenix or pandas (they are in requirements-dev.txt).
#
# Usage (from the repo root): python -m app.dev.phoenix_tools
import json
import sys

import pandas as pd
import phoenix as px
from llama_index.core import set_global_handler
from phoenix.client import Client

from app.router.router_agent import load_index_from_qdrant
from app.synthesizer.synthesizer_agent import answer_query


def launch_phoenix_tracing():
    # Local Phoenix dashboard that shows each router and synthesizer call
    print("Launching Arize Phoenix dashboard locally")
    session = px.launch_app()
    set_global_handler("arize_phoenix")
    return session


# Uploads every vector with its metadata so Phoenix can show the 3D projection
def upload_qdrant_dataset_with_vectors(index):
    try:
        qdrant_client = index.storage_context.vector_store.client
        points, _ = qdrant_client.scroll(collection_name="financial_filings", limit=3000, with_vectors=True)

        records = []
        for p in points:
            records.append({
                # LlamaIndex stores the chunk text inside the serialized node, not as a "text" key
                "text": json.loads(p.payload.get("_node_content", "{}")).get("text", ""),
                "company": p.payload.get("company", "N/A"),
                "year": p.payload.get("year", "N/A"),
                "section": p.payload.get("section", "N/A"),
                "vector": p.vector,
            })

        Client().datasets.create_dataset(
            name="financial_filings_vectors",
            dataframe=pd.DataFrame(records),
            input_keys=["text"],
            metadata_keys=["company", "year", "section"],
        )
        print("Vector dataset registered in Phoenix")
    except Exception as e:
        print(f"Dataset notice: {e}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to a legacy code page
    launch_phoenix_tracing()
    index = load_index_from_qdrant()
    upload_qdrant_dataset_with_vectors(index)

    print("\nInteractive Financial RAG session. Type a question, or 'exit' to quit.\n")
    try:
        while True:
            user_query = input("\n[Enter Question]: ").strip()
            if not user_query or user_query.lower() in ["exit", "quit"]:
                break
            answer, filters, _ = answer_query(user_query, index)
            print(f"\nFilters: {filters}\n\n{answer}")
    except KeyboardInterrupt:
        pass
    finally:
        print("Exiting RAG session.")
        px.close_app()


if __name__ == "__main__":
    main()
