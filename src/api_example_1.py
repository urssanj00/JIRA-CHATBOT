from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Corpus of documents
corpus = [
    "The Eiffel Tower is located in Paris.",
    "The Great Wall of China is visible from space.",
    "The Statue of Liberty is in New York."
]

# Encode corpus
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

def query_processor(query):
    # Query

    query_embedding = model.encode(query, convert_to_tensor=True)

    # Compute cosine similarity scores
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)

    # Find the most similar document
    top_result = cos_scores.argmax()
    ans = corpus[top_result]
    return ans

q = "Where is the Eiffel Tower?"
print(f"Query: {q}")
answer = query_processor(q)
print(f"Most similar document: {answer}")

q = "Where is the Great wall of China?"
print(f"Query: {q}")
answer = query_processor(q)
print(f"Most similar document: {answer}")

