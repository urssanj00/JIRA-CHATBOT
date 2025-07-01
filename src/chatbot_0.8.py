import pandas as pd
import spacy

import os
import json
import pickle
import numpy as np
 
from sentence_transformers import SentenceTransformer
import gradio as gr
from typing import Dict, List, Tuple, Any
import re
from datetime import datetime
import faiss
from collections import Counter
import threading
import time
from logger_config import logger
from logger_config import properties

nlp = spacy.load("en_core_web_sm")

class EnhancedJiraChatbot:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        logger.info("Initializing EnhancedJiraChatbot")
        self.encoder = SentenceTransformer(model_name)
        self.data_chunks = []
        self.embeddings = None
        self.index = None
        self.metadata = {}
        self.is_trained = False
        self.temperature = 0.7
        self.training_status = "Not started"
        self.model_save_dir = properties["model_path"]

    def preprocess_text(self, text: str) -> str:
        if pd.isna(text) or text == '':
            return ""
        text = str(text)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'!\S+\.\S+!', '', text)
        text = re.sub(r'[^\w\s\-\.\,\:\;]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def create_chunks(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        chunks = []
        for idx, row in df.iterrows():
            summary = f"Issue {row.get('Issue key', '')}: {self.preprocess_text(row.get('Summary', ''))}"
            chunks.append({
                'text': summary,
                'type': 'summary',
                'metadata': row.to_dict()
            })
            desc = self.preprocess_text(row.get('Description', ''))
            if desc:
                chunks.append({
                    'text': desc,
                    'type': 'description',
                    'metadata': row.to_dict()
                })
        return chunks

    def remove_sparse_columns(self, df: pd.DataFrame, threshold: int = 10) -> pd.DataFrame:
        non_null_counts = df.count()
        cols_to_remove = non_null_counts[non_null_counts <= threshold].index.tolist()
        logger.info(f"Columns removed (≤ {threshold} non-null records): {cols_to_remove}")
        return df.drop(columns=cols_to_remove)

    def train_from_csv(self, csv_path: str) -> bool:
        try:
            logger.info("Loading data...")
            df = pd.read_csv(csv_path)
            df = self.remove_sparse_columns(df)
            self.data_chunks = self.create_chunks(df)
            texts = [chunk['text'] for chunk in self.data_chunks]
            logger.info("Encoding embeddings...")
            self.embeddings = self.encoder.encode(texts, normalize_embeddings=True)
            # Build a simple index (brute-force for demo)
            self.index = self.embeddings
            self.df = df
            self.is_trained = True
            logger.info("Training complete.")
            return True
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.is_trained = False
            return False

    def analyze_query(self, query: str) -> Dict[str, Any]:
        doc = nlp(query)
        intent = "factual"
        if any(tok.lemma_ in ["count", "number", "how many", "total"] for tok in doc):
            intent = "count"
        elif any(tok.lemma_ in ["list", "show", "find", "filter", "get"] for tok in doc):
            intent = "filter"
        # Extract possible column and value
        col, val = None, None
        for ent in doc.ents:
            if ent.label_ in ["DATE"]:
                col, val = "Created", ent.text
        # Try to extract column and value from tokens
        for tok in doc:
            if tok.text.lower() in self.df.columns.str.lower():
                col = tok.text
        return {"intent": intent, "column": col, "value": val}

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        if not self.is_trained:
            logger.warning("Model not trained.")
            return []
        query_emb = self.encoder.encode([query], normalize_embeddings=True)[0]
        # Brute-force cosine similarity
        import numpy as np
        sims = np.dot(self.embeddings, query_emb)
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self.data_chunks[i]['text'], float(sims[i]), self.data_chunks[i]['metadata']) for i in top_idx]

    def generate_answer(self, query: str) -> str:
        if not self.is_trained:
            return "Model not trained."
        analysis = self.analyze_query(query)
        results = self.search(query)
        if analysis["intent"] == "count":
            return self._handle_count_query(analysis, results)
        elif analysis["intent"] == "filter":
            return self._handle_filter_query(analysis, results)
        else:
            return self._handle_factual_query(results)

    def _handle_count_query(self, analysis, results) -> str:
        col = analysis.get("column")
        val = analysis.get("value")
        if col and col in self.df.columns:
            if val:
                count = self.df[self.df[col].astype(str).str.contains(val, case=False, na=False)].shape[0]
                return f"There are {count} issues where {col} contains '{val}'."
            else:
                count = self.df[col].notna().sum()
                return f"There are {count} issues with a value in '{col}'."
        return f"Total issues: {self.df.shape[0]}"

    def _handle_filter_query(self, analysis, results) -> str:
        col = analysis.get("column")
        val = analysis.get("value")
        if col and val and col in self.df.columns:
            filtered = self.df[self.df[col].astype(str).str.contains(val, case=False, na=False)]
            if not filtered.empty:
                return f"Found {filtered.shape[0]} issues:\n" + "\n".join(filtered['Summary'].astype(str).head(5))
            else:
                return "No matching issues found."
        # Fallback: show top results
        return "Top relevant issues:\n" + "\n".join([r[0] for r in results[:5]])

    def _handle_factual_query(self, results) -> str:
        if results:
            return f"Most relevant info:\n{results[0][0]}"
        return "I couldn't find specific information to answer your query."

    def chat(self, message: str, history: List) -> Tuple[List, str]:
        if not message.strip():
            return history, ""
        try:
            answer = self.generate_answer(message)
            history = history + [[message, answer]]
            return history, ""
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return history, "Sorry, something went wrong."

# UI and other functions would be similar, but now the logic is more robust and dynamic.
def create_interface():
    """Create the main Gradio interface"""
    chatbot = EnhancedJiraChatbot()
    
    def train_model(csv_path, progress=gr.Progress()):
        """Training function with progress tracking"""
        if not csv_path:
            return "Please select a CSV file first.", "Not trained"
        
        def progress_callback(msg):
            progress(0.5, desc=msg)
        
        success = chatbot.train_from_csv(csv_path, progress_callback)
        
        if success:
            # Auto-save after successful training
            save_dir = os.path.join(os.path.dirname(csv_path), "trained_model")
            chatbot.save_model(save_dir)
            return f"Training completed! Model saved to: {save_dir}", "Trained ✓"
        else:
            return chatbot.training_status, "Training Failed ✗"
    
    def load_model(model_path):
        """Load pre-trained model"""
        if not model_path:
            return "Please select a model directory first.", "Not loaded"
        
        success = chatbot.load_model(model_path)
        if success:
            return "Model loaded successfully!", "Loaded ✓"
        else:
            return chatbot.training_status, "Load Failed ✗"
    
    def update_temperature(temp):
        """Update response temperature"""
        chatbot.temperature = temp
        return f"Temperature set to {temp}"
    
    # Create interface
    with gr.Blocks(title="Enhanced NLP Chatbot", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 Enhanced NLP Chatbot for Jira Data")
        
        with gr.Tab("🔧 Setup & Training"):
            gr.Markdown("## Step 1: Choose Training Data or Load Model")
            
            with gr.Row():
                with gr.Column():
                    csv_file = gr.File(label="Select CSV Training Data", file_types=[".csv"])
                    train_btn = gr.Button("Train Model", variant="primary", size="lg")
                    train_status = gr.Textbox(label="Training Status", interactive=False)
                    model_status = gr.Textbox(label="Model Status", value="Not trained", interactive=False)
                
                with gr.Column():
                    model_dir = gr.File(label="Or Load Pre-trained Model Directory", file_count="directory")
                    load_btn = gr.Button("Load Model", variant="secondary", size="lg")
                    
            gr.Markdown("## Step 2: Configure Settings")
            temperature_slider = gr.Slider(
                minimum=0.1, maximum=2.0, value=0.7, step=0.1,
                label="Response Temperature (Lower = More Focused, Higher = More Creative)"
            )
            temp_status = gr.Textbox(label="Current Temperature", value="0.7", interactive=False)
        
        with gr.Tab("💬 Chat"):
            gr.Markdown("## Ask Questions About Your Data")
            chatbot_interface = gr.Chatbot(height=500, label="Chat History")
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask about issues, counts, or specific tickets...", 
                    label="Your Question", scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            
            clear_btn = gr.Button("Clear Chat", variant="secondary")
            
            gr.Markdown("### 💡 Example Questions:")
            gr.Examples(
                examples=[
                    "How many bug issues are there?",
                    "What is the status of SRCTREEWIN-14221?",
                    "Show me high priority issues",
                    "List all issues in Short Term Backlog",
                    "What types of issues are most common?",
                    "What is the total number of issues in the dataset?",
                    "What is the total number of issues in month of March?",
                    "How many Critical Issues are raised in Month of March?",
                    "What are the different issue types present in the data?",
                    "List all unique project names and their corresponding project keys.",
                    "What are the different statuses an issue can have?",
                    "Can you provide the summary and description for the issue with the key \"SRCTREEWIN-14000\"?",
                    "What is the summary of the issue \"SRCTREEWIN-13894\" and what is the core problem described?"
                    "Explain the problem described in the issue \"SRCTREEWIN-13513\".",
                    "What is the error message for the issue \"SRCTREEWIN-12481\" related to \"Git not found\"?",
                    "Describe the issue where \"Closing a branch with one commit does not create a new commit\" (SRCTREEWIN-13668)." 
                    #"Filtering and Grouping",
                    "How many issues have the status \"Needs Triage\"?",
                    "List all issues that are categorized as \"Bug\". ",
                    "Which issues are related to \"OAuth\" problems? ",
                    "Group issues by \"Project key\" and count the number of issues in each project.",
                    "Identify issues where the \"Resolution\" is \"Fixed\". ",
                    "Which issues have \"Git\" as a component? ",
                    "List issues created by the reporter \"b2c3c286f465\". ",
                    #"Temporal Analysis",
                    "What is the creation date for the issue \"SRCTREEWIN-14000\"? ",
                    "Find the most recently updated issue in the dataset. ",
                    "How many issues were created in the year 2022? ",
                    "List all issues that have been resolved and their resolution dates. ",
                    "Advanced Analysis and Summarization",
                    "Summarize the common themes or problems identified across the \"Bug\" issue types. ",
                    "Are there any issues that seem to be duplicates or closely related based on their summaries or descriptions? (e.g., \"OAuth token keeps expiring\" and \"OAuth Fails, Unable to get Secret\"). ",
                    "Based on the descriptions, what are some of the technical details mentioned in the issues? ",
                    "What are the different versions affected by the reported bugs?" 
                ],
                inputs=msg
            )
        
        # Event handlers
        train_btn.click(train_model, inputs=[csv_file], outputs=[train_status, model_status])
        load_btn.click(load_model, inputs=[model_dir], outputs=[train_status, model_status])
        temperature_slider.change(update_temperature, inputs=[temperature_slider], outputs=[temp_status])
        
        msg.submit(chatbot.chat, [msg, chatbot_interface], [chatbot_interface, msg])
        send_btn.click(chatbot.chat, [msg, chatbot_interface], [chatbot_interface, msg])
        clear_btn.click(lambda: [], None, chatbot_interface)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, debug=True, server_name="0.0.0.0", server_port=7860)    