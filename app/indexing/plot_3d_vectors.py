# HELPER FUNCTİOBN FOR BUILDING NODES IN 3D
# Dev-only script (needs requirements-dev.txt). Run from the repo root: python -m app.indexing.plot_3d_vectors

import json
import sys

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from app.router.router_agent import load_index_from_qdrant

# The sections most questions are about get their own color; the rest are grouped as "Other items"
MAIN_SECTIONS = {"Item 1": "Business (Item 1)", "Item 1A": "Risk Factors (Item 1A)", "Item 7": "MD&A (Item 7)",
                 "Item 8": "Financial Statements (Item 8)"}


def section_label(payload: dict) -> str:
    # NVIDIA keeps its financial statements in Item 15; for the others Item 15 only lists exhibits
    if payload.get("section") == "Item 15" and payload.get("company") == "NVDA":
        return MAIN_SECTIONS["Item 8"]
    return MAIN_SECTIONS.get(payload.get("section"), "Other items")


def load_points():
    index = load_index_from_qdrant()
    qdrant_client = index.storage_context.vector_store.client
    # The limit must be above the number of chunks (3,913 for the 12 filings), or companies go missing
    points, _ = qdrant_client.scroll(collection_name="financial_filings", limit=100_000, with_vectors=True)
    return points


def build_figure(points):
    vectors = np.array([p.vector for p in points])
    # Applying PCA reduction to project the 1024-dimension embeddings to 3D
    coords = PCA(n_components=3).fit_transform(vectors)
    df = pd.DataFrame({
        "PCA_1": coords[:, 0],
        "PCA_2": coords[:, 1],
        "PCA_3": coords[:, 2],
        "Company": [str(p.payload.get("company", "Unknown")) for p in points],
        "Section": [section_label(p.payload) for p in points],
        "Year": [str(p.payload.get("year", "Unknown")) for p in points],
        # LlamaIndex stores the chunk text inside the serialized node (_node_content), not as a "text" key
        "Snippet": [json.loads(p.payload.get("_node_content", "{}")).get("text", "")[:120].replace("\n", " ") + "..."
                    for p in points],
    })
    companies = sorted(df["Company"].unique())
    fig = px.scatter_3d(
        df,
        x="PCA_1",
        y="PCA_2",
        z="PCA_3",
        # Color shows the 10-K section; with six companies, symbols per company would repeat
        # (3D plots have only five marker shapes), so the company is shown on hover
        color="Section",
        category_orders={"Section": list(MAIN_SECTIONS.values()) + ["Other items"]},
        hover_data=["Company", "Year", "Section", "Snippet"],
        title=f"3D Vector Space of SEC 10-K Filings ({', '.join(companies)}, FY2024-2025)",
        template="plotly_dark",
        opacity=0.8,
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        scene=dict(
            xaxis_title="Vector PC 1",
            yaxis_title="Vector PC 2",
            zaxis_title="Vector PC 3"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    print("Connecting to Qdrant Database & Fetching Vector Embeddings...")
    points = load_points()
    print(f"Loaded {len(points)} vector points. Calculating 3D PCA Projection...")
    build_figure(points).show()
