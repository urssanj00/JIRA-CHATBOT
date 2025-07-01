import os
import requests
import json
from time import sleep

# === CONFIGURATION ===
PROJECT_KEY = "SPARK"
TOTAL_ISSUES = 5000
BATCH_SIZE = 1000  # JIRA API limit
SAVE_FOLDER = r"D:\VNIT_Mtech\Subjects\11.NLP\NLP_project\jira_spark_data"  
# =====================

os.makedirs(SAVE_FOLDER, exist_ok=True)
all_issues = []

# Step 1: Paginated fetch from JIRA
print(f"🔄 Fetching up to {TOTAL_ISSUES} issues from Apache JIRA...")
for start_at in range(0, TOTAL_ISSUES, BATCH_SIZE):
    print(f"🔹 Fetching issues {start_at + 1} to {start_at + BATCH_SIZE}...")
    url = "https://issues.apache.org/jira/rest/api/2/search"
    params = {
        "jql": f"project={PROJECT_KEY}",
        "startAt": start_at,
        "maxResults": BATCH_SIZE,
        "fields": "summary,description,comment,issuetype,status,created,updated,priority,reporter,assignee,labels,components,fixVersions,versions,resolution"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    all_issues.extend(data.get("issues", []))
    sleep(1)  # Be polite to JIRA API

# Save raw data (optional)
raw_path = os.path.join(SAVE_FOLDER, f"{PROJECT_KEY}_jira_issues_raw.json")
with open(raw_path, "w", encoding="utf-8") as f:
    json.dump(all_issues, f, indent=2)
print(f"✅ Raw data saved to: {raw_path}")

# Step 2: Format for LLM chatbot training
chat_data = []
for issue in all_issues:
    fields = issue["fields"]
    def get(field, sub=None):
        try:
            val = fields.get(field)
            if sub and isinstance(val, dict):
                return val.get(sub, "")
            if isinstance(val, list):
                return ", ".join([v.get(sub, "") for v in val if sub in v])
            return val or ""
        except: return ""

    # Extract relevant fields
    key = issue.get("key", "")
    summary = get("summary")
    description = get("description") or "No description."
    status = get("status", "name")
    priority = get("priority", "name")
    issue_type = get("issuetype", "name")
    created = get("created")
    updated = get("updated")
    reporter = get("reporter", "displayName")
    assignee = get("assignee", "displayName") or "Unassigned"
    resolution = get("resolution", "name") or "Unresolved"
    labels = get("labels")
    components = get("components", "name")
    versions = get("versions", "name")
    fix_versions = get("fixVersions", "name")
    
    comments = fields.get("comment", {}).get("comments", [])
    comment_texts = "\n".join([f"- {c['author']['displayName']}: {c['body']}" for c in comments[:5]])  # limit to 5

    # Instruction/input/output format
    input_text = f"""Issue ID: {key}
Type: {issue_type}
Summary: {summary}
Status: {status}, Priority: {priority}
Created: {created}, Updated: {updated}
Reporter: {reporter}, Assignee: {assignee}
Resolution: {resolution}
Labels: {labels}
Components: {components}
Affects Versions: {versions}
Fix Versions: {fix_versions}

Description:
{description}

Top Comments:
{comment_texts}
"""

    output_text = f"Issue {key} is a {priority} {issue_type} that is currently '{status}'. It was reported by {reporter} and is assigned to {assignee}. Summary: {summary}. Description: {description[:300]}..."

    chat_data.append({
        "instruction": "Summarize the following software issue for support or triage:",
        "input": input_text,
        "output": output_text
    })

# Save chatbot training dataset
final_path = os.path.join(SAVE_FOLDER, f"{PROJECT_KEY}_chatbot_training_data.json")
with open(final_path, "w", encoding="utf-8") as f:
    json.dump(chat_data, f, indent=2)

print(f" Chatbot training data saved to: {final_path}")
print(f" Total samples: {len(chat_data)}")