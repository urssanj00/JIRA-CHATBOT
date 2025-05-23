from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import threading

app = Flask(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
#CORPUS_FILE = "information_doc.txt"
CORPUS_FILE = r"C:\Sanjeev\VNIT_CLASSES\NLP_PROJ\dataset\output.json"

# Globals
model = SentenceTransformer(MODEL_NAME)
corpus = []
corpus_embeddings = None
lock = threading.Lock()

def load_corpus_and_embeddings():
    global corpus, corpus_embeddings
    with lock:
        with open(CORPUS_FILE, "r", encoding="utf-8") as file:
            corpus = [line.strip() for line in file if line.strip()]
        corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
        print(f"Corpus reloaded: {len(corpus)} documents.")

# Initial load
load_corpus_and_embeddings()

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query_text = data.get("query", "")
    if not query_text:
        return jsonify({"error": "Query text is required."}), 400

    with lock:
        query_embedding = model.encode(query_text, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)
        top_result = int(cos_scores.argmax())
        answer = corpus[top_result]

    return jsonify({
        "query": query_text,
        "most_similar": answer
    })

@app.route('/reload', methods=['POST'])
def reload_corpus():
    load_corpus_and_embeddings()
    return jsonify({"status": "Corpus reloaded.", "num_documents": len(corpus)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
