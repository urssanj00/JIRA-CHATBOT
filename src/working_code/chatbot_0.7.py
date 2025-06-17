
"""
Environment: nlp_chat ;python=3.12.10
Enhanced NLP Chatbot with User Interface and Training Options
============================================================
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import gradio as gr
from typing import Dict, List, Tuple, Any
import re
from datetime import datetime
import faiss
from collections import Counter
import threading
import time

class EnhancedJiraChatbot:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # Using a better encoder model for improved embeddings
        self.encoder = SentenceTransformer(model_name)
        self.data_chunks = []
        self.embeddings = None
        self.index = None
        self.metadata = {}
        self.is_trained = False
        self.temperature = 0.7
        self.training_status = "Not started"
        
        # Better reflection patterns for query understanding
        self.reflection_patterns = {
            'count_keywords': ['how many', 'count', 'number of', 'total', 'sum'],
            'statistical_keywords': ['average', 'mean', 'median', 'percentage', 'ratio'],
            'filter_keywords': ['show me', 'list', 'find', 'get', 'filter', 'status of'],
            'factual_keywords': ['what is', 'who is', 'when was', 'where is', 'why']
        }
    
    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing with better cleaning"""
        if pd.isna(text) or text == '':
            return ""
        
        text = str(text)
        # Clean URLs, special characters, and normalize whitespace
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'!\S+\.\S+!', '', text)
        text = re.sub(r'[^\w\s\-\.\,\:\;]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def create_smart_chunks(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create intelligent chunks from DataFrame"""
        chunks = []
        
        for idx, row in df.iterrows():
            # Summary chunk for quick lookups
            summary_text = f"Issue {row.get('Issue key', '')}: {self.preprocess_text(row.get('Summary', ''))}"
            summary_chunk = {
                'text': summary_text,
                'type': 'summary',
                'metadata': {
                    'issue_key': row.get('Issue key', ''),
                    'issue_type': row.get('Issue Type', ''),
                    'status': row.get('Status', ''),
                    'priority': row.get('Priority', ''),
                    'component': row.get('Component/s', ''),
                    'assignee': row.get('Assignee', ''),
                    'reporter': row.get('Reporter', '')
                }
            }
            chunks.append(summary_chunk)
            
            # Detail chunk for complex queries
            if pd.notna(row.get('Description')) and len(str(row.get('Description', ''))) > 50:
                detail_text = f"Details for {row.get('Issue key', '')}: {self.preprocess_text(row.get('Description', ''))}"
                detail_chunk = {
                    'text': detail_text,
                    'type': 'detail',
                    'metadata': summary_chunk['metadata'].copy()
                }
                chunks.append(detail_chunk)
            
            # Categorical chunk for statistics
            categorical_text = f"{row.get('Issue Type', '')} {row.get('Priority', '')} {row.get('Status', '')} {row.get('Component/s', '')}"
            categorical_chunk = {
                'text': categorical_text,
                'type': 'categorical',
                'metadata': summary_chunk['metadata'].copy()
            }
            chunks.append(categorical_chunk)
        
        return chunks
    
    def train_from_csv(self, csv_path: str, progress_callback=None) -> bool:
        """Train the chatbot from CSV file with progress tracking"""
        try:
            self.training_status = "Loading data..."
            if progress_callback:
                progress_callback("Loading CSV file...")
            
            # Load CSV
            df = pd.read_csv(csv_path)
            
            self.training_status = "Creating chunks..."
            if progress_callback:
                progress_callback(f"Processing {len(df)} rows...")
            
            # Create chunks
            self.data_chunks = self.create_smart_chunks(df)
            
            self.training_status = "Generating embeddings..."
            if progress_callback:
                progress_callback(f"Generating embeddings for {len(self.data_chunks)} chunks...")
            
            # Generate embeddings
            texts = [chunk['text'] for chunk in self.data_chunks]
            self.embeddings = self.encoder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
            
            self.training_status = "Building search index..."
            if progress_callback:
                progress_callback("Building FAISS index...")
            
            # Build FAISS index
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(self.embeddings)
            
            # Create metadata
            self.metadata = {
                'total_chunks': len(self.data_chunks),
                'total_issues': len(df),
                'issue_types': df['Issue Type'].value_counts().to_dict() if 'Issue Type' in df.columns else {},
                'statuses': df['Status'].value_counts().to_dict() if 'Status' in df.columns else {},
                'priorities': df['Priority'].value_counts().to_dict() if 'Priority' in df.columns else {},
                'trained_at': datetime.now().isoformat()
            }
            
            self.is_trained = True
            self.training_status = "Training completed successfully!"
            if progress_callback:
                progress_callback("Training completed successfully!")
            
            return True
            
        except Exception as e:
            self.training_status = f"Training failed: {str(e)}"
            if progress_callback:
                progress_callback(f"Training failed: {str(e)}")
            return False
    
    def save_model(self, output_dir: str) -> bool:
        """Save trained model to directory"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save components
            with open(os.path.join(output_dir, 'chunks.pkl'), 'wb') as f:
                pickle.dump(self.data_chunks, f)
            
            np.save(os.path.join(output_dir, 'embeddings.npy'), self.embeddings)
            
            with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
                pickle.dump(self.metadata, f)
            
            return True
        except Exception as e:
            print(f"Save failed: {e}")
            return False
    
    def load_model(self, model_dir: str) -> bool:
        """Load pre-trained model"""
        try:
            # Load components
            with open(os.path.join(model_dir, 'chunks.pkl'), 'rb') as f:
                self.data_chunks = pickle.load(f)
            
            self.embeddings = np.load(os.path.join(model_dir, 'embeddings.npy'))
            
            with open(os.path.join(model_dir, 'metadata.pkl'), 'rb') as f:
                self.metadata = pickle.load(f)
            
            # Rebuild index
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(self.embeddings)
            
            self.is_trained = True
            self.training_status = "Model loaded successfully!"
            return True
            
        except Exception as e:
            self.training_status = f"Load failed: {str(e)}"
            return False
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine response strategy"""
        query_lower = query.lower()
        return {
            'is_count': any(kw in query_lower for kw in self.reflection_patterns['count_keywords']),
            'is_statistical': any(kw in query_lower for kw in self.reflection_patterns['statistical_keywords']),
            'is_filter': any(kw in query_lower for kw in self.reflection_patterns['filter_keywords']),
            'is_factual': any(kw in query_lower for kw in self.reflection_patterns['factual_keywords'])
        }
    
    def search_with_temperature(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Enhanced search with temperature-based ranking"""
        if not self.is_trained:
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query], normalize_embeddings=True)
        
        # Search with more candidates
        distances, indices = self.index.search(query_embedding, min(top_k * 3, len(self.data_chunks)))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.data_chunks):
                # Apply temperature to distance scoring
                score = distances[0][i]
                adjusted_score = score / self.temperature if self.temperature > 0 else score
                
                chunk = self.data_chunks[idx]
                results.append((chunk['text'], adjusted_score, chunk.get('metadata', {})))
        
        # Sort by adjusted score and return top_k
        results.sort(key=lambda x: x[1])
        return results[:top_k]
    
    def generate_answer(self, query: str) -> str:
        """Generate answer with improved logic"""
        if not self.is_trained:
            return "Please train the model first by uploading a CSV file."
        
        # Analyze query
        query_analysis = self.analyze_query(query)
        
        # Search for relevant content
        search_results = self.search_with_temperature(query)
        
        if not search_results:
            return "I couldn't find relevant information for your query. Please try rephrasing your question."
        
        # Handle different query types
        if query_analysis['is_count']:
            return self._handle_count_query(query, search_results)
        elif query_analysis['is_filter']:
            return self._handle_filter_query(query, search_results)
        else:
            return self._handle_general_query(query, search_results)
    
    def _handle_count_query(self, query: str, results: List) -> str:
        """Handle counting queries with statistics"""
        query_lower = query.lower()
        
        # Extract category to count
        if 'bug' in query_lower:
            count = self.metadata.get('issue_types', {}).get('Bug', 0)
            return f"There are {count} bug issues in the dataset."
        elif 'issue' in query_lower and 'type' in query_lower:
            types = self.metadata.get('issue_types', {})
            return f"Issue type distribution: {', '.join([f'{k}: {v}' for k, v in types.items()])}"
        elif 'status' in query_lower:
            statuses = self.metadata.get('statuses', {})
            return f"Status distribution: {', '.join([f'{k}: {v}' for k, v in statuses.items()])}"
        else:
            return f"Total issues in dataset: {self.metadata.get('total_issues', 0)}"
    
    def _handle_filter_query(self, query: str, results: List) -> str:
        """Handle filtering/listing queries"""
        filtered_results = []
        for text, score, metadata in results[:10]:  # Show more results for filtering
            if metadata:
                issue_key = metadata.get('issue_key', '')
                status = metadata.get('status', '')
                priority = metadata.get('priority', '')
                filtered_results.append(f"• {issue_key}: {text[:100]}... (Status: {status}, Priority: {priority})")
        
        if filtered_results:
            return f"Found {len(filtered_results)} relevant issues:\n\n" + "\n".join(filtered_results)
        else:
            return "No matching issues found for your filter criteria."
    
    def _handle_general_query(self, query: str, results: List) -> str:
        """Handle general factual queries"""
        # Use top result for specific factual queries
        if results:
            text, score, metadata = results[0]
            if metadata and metadata.get('issue_key'):
                issue_key = metadata['issue_key']
                status = metadata.get('status', 'Unknown')
                priority = metadata.get('priority', 'Unknown')
                return f"Issue {issue_key}:\n{text}\n\nStatus: {status} | Priority: {priority}"
            else:
                return f"Based on the dataset: {text}"
        
        return "I couldn't find specific information to answer your query."
    
    def chat(self, message: str, history: List) -> Tuple[List, str]:
        """Main chat interface"""
        if not message.strip():
            return history, ""
        
        try:
            answer = self.generate_answer(message)
            history.append([message, answer])
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            history.append([message, error_msg])
        
        return history, ""

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
                    "What types of issues are most common?"
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