from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

sentences = [
    "Apple Inc. reported quarterly revenue of $89.5 billion.",
    "Microsoft Corporation announced cloud revenue growth.",
    "The baseball game was delayed by heavy rain."
]

embeddings = model.encode(sentences)

print(f"Number of sentences encoded: {len(embeddings)}")
print(f"Embedding vector dimension: {len(embeddings[0])}")

sim_apple_msft = util.cos_sim(embeddings[0], embeddings[1])
sim_apple_baseball = util.cos_sim(embeddings[0], embeddings[2])

print(f"Similarity (Apple vs Microsoft): {sim_apple_msft.item():.4f}")
print(f"Similarity (Apple vs Baseball): {sim_apple_baseball.item():.4f}")