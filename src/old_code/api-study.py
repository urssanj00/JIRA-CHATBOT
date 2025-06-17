from sentence_transformers import SentenceTransformer

# Load a pretrained model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sentences to encode
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# Generate embeddings
embeddings = model.encode(sentences)

print(embeddings.shape)  # e.g., (3, 384)
