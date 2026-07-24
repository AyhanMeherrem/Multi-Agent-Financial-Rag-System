from qdrant_client import QdrantClient

client = QdrantClient(path="./data/qdrant_db")

# 1. Print Collection Statistics
info = client.get_collection("financial_filings")
print(f"Total Stored Vectors: {info.points_count}")
print(f"Vector Dimension: {info.config.params.vectors.size}")

# 2. Inspect Sample Vector Point & Payload Metadata
points, _ = client.scroll(collection_name="financial_filings", limit=1)
sample_point = points[0]

print(f"Point ID: {sample_point.id}")
print(f"Metadata Payload: {sample_point.payload}")

client.close()

# Expected output:
# Total Stored Vectors: 2146
# Vector Dimension: 1024
# Point ID: <uuid>
# Metadata Payload: {'company': 'AAPL', 'year': '2024', 'section': 'Item 8', '_node_content': '{"id_": "<uuid>", ...}'}