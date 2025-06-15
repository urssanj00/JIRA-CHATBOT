from flask import Flask, request, jsonify
import json
import re
from sentence_transformers import SentenceTransformer, util
import threading

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

app = Flask(__name__)

issues = []
issues_lock = threading.Lock()

filename =  r"C:\Sanjeev\VNIT_CLASSES\NLP_PROJ\dataset\output.json"

def load_issues():
    global issues
    with issues_lock:
        with open(filename, 'r', encoding='utf-8') as f:
            issues = json.load(f)
        print(f"Loaded {len(issues)} issues.")

# Initial load
load_issues()

def extract_issue_key(query):
    print(f'query : {query}')
    match = re.search(r'([A-Z]+-\d+)', query)
    print(f'match : {match}')
    return match.group(1) if match else None

def extract_field_name(query):
    # Try to extract the field requested (e.g. "description", "summary", etc.)
    # Looks for phrases like "the <field> of Issue" or "for Issue"
    match = re.search(r'(?:the|show|give|provide|what is|who is|list|display)\s+([\w\s/-]+?)(?:\s+of|\s+for)?\s+Issue', query, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # fallback: last word before 'of Issue'
    match = re.search(r'([\w\s/-]+)\s+of Issue', query, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def semantic_field_match(requested_field, issue_dict):

    # Get all field names from the issue
    field_names = list(issue_dict.keys())
    # Encode the requested field and all actual field names
    embeddings = model.encode([requested_field] + field_names)
    query_emb = embeddings[0]
    field_embs = embeddings[1:]
    # Compute semantic similarity
    scores = util.cos_sim(query_emb, field_embs)[0]
    best_idx = int(scores.argmax())
    return field_names[best_idx], float(scores[best_idx])

@app.route('/query', methods=['POST'])
def query():
    print(0)
    data = request.get_json()
    print(1)
    query_text = data.get("query", "")
    print(2)

    issue_key = extract_issue_key(query_text)
    print(3)

    requested_field = extract_field_name(query_text)
    print(4)

    if not issue_key or not requested_field:
        return jsonify({"error": "Could not extract issue key or field from query.", "query": query_text}), 400
    print(5)

    with issues_lock:
        for issue in issues:
            if issue.get("Issue key", "").upper() == issue_key.upper():
                field_name, score = semantic_field_match(requested_field, issue)
                value = issue.get(field_name, "Not found")
                return jsonify({
                    "issue_key": issue_key,
                    "matched_field": field_name,
                    "similarity": round(score, 3),
                    "value": value,
                    "query": query_text
                })

    return jsonify({"error": f"Issue {issue_key} not found.", "query": query_text}), 404

@app.route('/reload', methods=['POST'])
def reload_corpus():
    load_issues()
    with issues_lock:
        num_documents = len(issues)
    return jsonify({"status": "Corpus reloaded.", "num_documents": num_documents})

if __name__ == '__main__':
    app.run(debug=True)
