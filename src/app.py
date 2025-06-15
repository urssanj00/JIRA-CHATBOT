from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch
import json
import re
app = Flask(__name__)

filename =  r"C:\Sanjeev\VNIT_CLASSES\NLP_PROJ\dataset\output.json"

# Load data
with open(filename, "r", encoding="utf-8") as f:
    issues = json.load(f)

# Load models
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # fast & small
llm = pipeline("text-generation", model="distilgpt2")  # optional

# Prepare embeddings
documents = [f"{i['Summary']} - {i.get('Description', '')}" for i in issues]
document_embeddings = embedder.encode(documents, convert_to_tensor=True)

@app.route("/chat", methods=["POST"])
def chat():
    #print(f'llm : {llm}')
    #print(f'document_embeddings : {document_embeddings}')
    user_query = request.json.get("query", "")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # Regex to extract status filter: e.g., "Status = 'Needs Triage'"
    status_match = re.search(r"status\s*=\s*'([^']+)'", user_query, re.IGNORECASE)

    if status_match:
        status_filter = status_match.group(1).lower()
        filtered_issues = [i for i in issues if i.get("Status", "").lower() == status_filter]
    else:
        filtered_issues = issues

    # If nothing matches the filter
    if not filtered_issues:
        return jsonify({"error": f"No issues found with status = '{status_match.group(1)}'" if status_match else "No matching issues found."}), 404

    # Check for "find all" keyword
    if "find all" in user_query.lower():
        results = []
        for issue in filtered_issues:
            prompt = f"User: {user_query}\nIssue: {issue.get('Description', '')}\nAnswer:"
            llm_response = llm(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
            results.append({
                "issue_key": issue.get("Issue key"),
                "summary": issue.get("Summary"),
                "description": issue.get("Description"),
                "llm_response": llm_response
            })
        return jsonify(results)

    # Default: return most relevant match using embeddings
    documents = [f"{i['Summary']} - {i.get('Description', '')}" for i in filtered_issues]
    document_embeddings = embedder.encode(documents, convert_to_tensor=True)
    query_embedding = embedder.encode(user_query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
    top_score, top_idx = torch.topk(scores, k=1)
    idx = top_idx[0].item()
    score = top_score[0].item()

    match = filtered_issues[idx]
    match_text = match.get("Description", "")

    # Optional: Generate LLM response
    prompt = f"User: {user_query}\nIssue: {match_text}\nAnswer:"
    prompt = prompt[:1000]  # trim to 1000 characters if needed
    print(f'{prompt}')
    llm_response = llm(prompt, max_new_tokens=100, num_return_sequences=1)[0]['generated_text']

    return jsonify({
        "match_score": round(score, 4),
        "issue_key": match.get("Issue key"),
        "summary": match.get("Summary"),
        "description": match_text,
        "llm_response": llm_response
    })

if __name__ == "__main__":
    app.run(debug=True)
