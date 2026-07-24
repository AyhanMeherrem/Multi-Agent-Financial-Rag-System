# HELPER FUNCTİOBN FOR BUILDING NODES IN 3D

import os
import sys
sys.path.append(".")
sys.stdout.reconfigure(encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from app.router.router_agent import load_index_from_qdrant

print("Connecting to Qdrant Database & Fetching 2,146 Vector Embeddings...")
index = load_index_from_qdrant()
qdrant_client = index.storage_context.vector_store.client

points, _ = qdrant_client.scroll(collection_name="financial_filings", limit=3000, with_vectors=True)

print(f"Loaded {len(points)} vector points. Calculating 3D PCA Projection...")

vectors = np.array([p.vector for p in points])
companies = [str(p.payload.get("company", "Unknown")) for p in points]
sections = [str(p.payload.get("section", "Unknown")) for p in points]
years = [str(p.payload.get("year", "Unknown")) for p in points]
snippets = [str(p.payload.get("text", ""))[:120].replace("\n", " ") + "..." for p in points]

# Applying PCA reductio n to convert it into 
pca = PCA(n_components=3)
coords = pca.fit_transform(vectors)

df = pd.DataFrame({
    "PCA_1": coords[:, 0],
    "PCA_2": coords[:, 1],
    "PCA_3": coords[:, 2],
    "Company": companies,
    "Section": sections,
    "Year": years,
    "Snippet": snippets
})

fig = px.scatter_3d(
    df,
    x="PCA_1",
    y="PCA_2",
    z="PCA_3",
    color="Section",
    symbol="Company",
    hover_data=["Company", "Year", "Section", "Snippet"],
    title="Interactive 3D Vector Space Map — SEC 10-K Filings (Apple vs Microsoft)",
    template="plotly_dark",
    opacity=0.85
)

fig.update_layout(
    scene=dict(
        xaxis_title="Vector PC 1",
        yaxis_title="Vector PC 2",
        zaxis_title="Vector PC 3"
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.show()
