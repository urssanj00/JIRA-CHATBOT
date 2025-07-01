"""
Llama 2 7B Fine-tuned Jira Chatbot with JSON to Pickle Training

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
import requests
from tqdm import tqdm
from llama_cpp import Llama
import warnings
import gc
warnings.filterwarnings("ignore")

class LocalModelManager:
    """Manages Llama 2 7B model download and initialization"""
    
    def __init__(self):
        self.model_dir = "/Users/sanjeev/VNIT/JIRA-CHATBOT/models"
        self.model_name = "llama-2-7b-chat.Q4_K_M.gguf"
        self.model_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
        self.model_path = os.path.join(self.model_dir, self.model_name)
        
    def download_model(self, progress_callback=None):
        """Download Llama 2 7B model if not present"""
        if os.path.exists(self.model_path):
            print(f"Model already exists at {self.model_path}")
            return True
            
        os.makedirs(self.model_dir, exist_ok=True)
        
        try:
            print(f"Downloading Llama 2 7B from {self.model_url}")
            if progress_callback:
                progress_callback("Downloading Llama 2 7B model...")
            
            response = requests.get(self.model_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(self.model_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"Llama 2 7B downloaded successfully to {self.model_path}")
            return True
            
        except Exception as e:
            print(f"Error downloading model: {e}")
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
            return False
    
    def load_model(self):
        """Load Llama 2 7B with optimized settings"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            file_size = os.path.getsize(self.model_path)
            print(f"Loading Llama 2 7B from: {self.model_path} (Size: {file_size / (1024**3):.2f} GB)")
            
            model = Llama(
                model_path=self.model_path,
                n_ctx=4096,
                n_batch=512,
                n_threads=4,
                n_gpu_layers=0,
                use_mlock=True,
                use_mmap=True,
                verbose=False,
                f16_kv=True,
            )
            print("Llama 2 7B loaded successfully!")
            return model
            
        except Exception as e:
            print(f"Error loading Llama 2 7B: {e}")
            return None

class FineTunedJiraChatbot:
    def __init__(self):
        self.model = None
        self.model_manager = LocalModelManager()
        self.embeddings = None
        self.embedding_model = None
        self.data = None
        self.index = None
        self.temperature = 0.7
        self.training_status = "Not trained"
        self.max_length = 1024
        self.model_loaded = False
        self.generation_lock = threading.Lock()
        self.stats_cache = {}
        self.search_params = {}  # Initialize search_params
        
        try:
            print("Loading embedding model...")
            self.embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
            print("Embedding model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")
            self.embedding_model = None
    
    def initialize_model(self, progress_callback=None):
        """Initialize Llama 2 7B model"""
        try:
            if progress_callback:
                progress_callback("Checking for Llama 2 7B...")
            
            if not self.model_manager.download_model(progress_callback):
                return False
            
            if progress_callback:
                progress_callback("Loading Llama 2 7B...")
            
            self.model = self.model_manager.load_model()
            
            if self.model:
                self.model_loaded = True
                self.training_status = "Llama 2 7B loaded successfully"
                return True
            else:
                self.training_status = "Failed to load Llama 2 7B"
                return False
                
        except Exception as e:
            self.training_status = f"Model initialization failed: {str(e)}"
            return False
    
    def json_to_pickle(self, json_path: str, progress_callback=None) -> str:
        """Convert JSON data to pickle format and process for training"""
        try:
            if progress_callback:
                progress_callback("Loading JSON data...")
            
            print(f"Loading JSON from: {json_path}")
            
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            print(f"Loaded {len(json_data) if isinstance(json_data, list) else 1} items from JSON")
            
            if progress_callback:
                progress_callback("Processing JSON data...")
            
            # Process JSON data into training format
            processed_data = self.process_json_data(json_data)
            
            # Create pickle file path
            base_name = os.path.splitext(os.path.basename(json_path))[0]
            pickle_dir = os.path.dirname(json_path)
            pickle_path = os.path.join(pickle_dir, f"{base_name}_processed.pkl")
            
            if progress_callback:
                progress_callback("Saving as pickle...")
            
            print(f"Saving pickle to: {pickle_path}")
            
            with open(pickle_path, 'wb') as f:
                pickle.dump(processed_data, f)
            
            print(f"Pickle file created successfully at: {pickle_path}")
            print(f"Processed {len(processed_data)} items")
            
            return pickle_path
            
        except Exception as e:
            error_msg = f"JSON to pickle conversion failed: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
    
    def process_json_data(self, json_data) -> List[Dict]:  # Removed type hint that was causing issues
        """Process JSON data into standardized format"""
        processed = []
        
        # Handle both list and single dict formats
        data_items = json_data if isinstance(json_data, list) else [json_data]
        
        for item in data_items:
            # Handle different JSON formats
            if isinstance(item, dict):
                # Extract key fields
                issue_id = item.get('key', item.get('id', f"ISSUE-{len(processed)}"))
                summary = item.get('summary', item.get('title', ''))
                description = item.get('description', item.get('desc', ''))
                status = item.get('status', 'Unknown')
                priority = item.get('priority', 'Medium')
                issue_type = item.get('type', item.get('issuetype', 'Task'))
                
                # Create input/output format
                input_text = f"""Issue ID: {issue_id}
Type: {issue_type}
Summary: {summary}
Status: {status}
Priority: {priority}
Description: {description}"""
                
                output_text = f"This is {issue_type} {issue_id}: {summary}. Status: {status}, Priority: {priority}."
                
                processed.append({
                    'input': input_text,
                    'output': output_text,
                    'issue_id': issue_id,
                    'type': issue_type,
                    'summary': summary,
                    'status': status,
                    'priority': priority,
                    'description': description
                })
        
        return processed
    
    def train_from_json(self, json_path: str, progress_callback=None) -> Tuple[bool, str]:
        """Train from JSON data by converting to pickle first"""
        try:
            # Convert JSON to pickle
            pickle_path = self.json_to_pickle(json_path, progress_callback)
            
            # Load the pickle data
            success = self.train_from_pickle(pickle_path, progress_callback)
            
            if success:
                self.training_status = f"Training completed! Pickle saved at: {pickle_path}"
            
            return success, pickle_path
            
        except Exception as e:
            self.training_status = f"JSON training failed: {str(e)}"
            return False, ""
    
    def train_from_pickle(self, pickle_path: str, progress_callback=None) -> bool:
        """Load and process pickle data"""
        try:
            if progress_callback:
                progress_callback("Loading pickle data...")
            
            with open(pickle_path, 'rb') as f:
                pickle_data = pickle.load(f)
            
            chunks = self.create_chunks_from_pickle(pickle_data)
            self.data = chunks
            
            if progress_callback:
                progress_callback("Building embeddings...")
            
            # Build embeddings for semantic search
            if self.embedding_model and chunks:
                self.embeddings = self.build_embeddings(chunks)
                if self.embeddings is not None:
                    self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
                    self.index.add(self.embeddings.astype('float32'))
                    
                if progress_callback:
                    progress_callback("Optimizing search parameters...")
                self.search_params = self.optimize_search_parameters()

            if not self.model_loaded:
                if not self.initialize_model(progress_callback):
                    return False
            
            self.training_status = f"Data loaded: {len(chunks)} issues"
            return True
            
        except Exception as e:
            self.training_status = f"Pickle loading failed: {str(e)}"
            return False
    
    def create_chunks_from_pickle(self, pickle_data) -> List[Dict]:
        """Create chunks from pickle data and compute statistics"""
        chunks = []
        
        if isinstance(pickle_data, list):
            data = pickle_data
        elif isinstance(pickle_data, dict) and 'data' in pickle_data:
            data = pickle_data['data']
        else:
            data = [pickle_data] if pickle_data else []
        
        type_counts = Counter()
        status_counts = Counter()
        priority_counts = Counter()
        
        for item in data:
            if isinstance(item, dict):
                issue_id = item.get('issue_id', item.get('key', 'UNKNOWN'))
                issue_type = item.get('type', 'Task')
                summary = item.get('summary', '')
                status = item.get('status', 'Unknown')
                priority = item.get('priority', 'Medium')
                description = item.get('description', '')
                
                type_counts[issue_type] += 1
                status_counts[status] += 1
                priority_counts[priority] += 1
                
                chunk = {
                    'issue_id': issue_id,
                    'type': issue_type,
                    'summary': summary,
                    'status': status,
                    'priority': priority,
                    'description': description,
                    'full_text': item.get('input', ''),
                    'output': item.get('output', ''),
                    'original_data': item
                }
                chunks.append(chunk)
        
        self.stats_cache = {
            'type_counts': type_counts,
            'status_counts': status_counts,
            'priority_counts': priority_counts,
            'total_count': len(chunks)
        }
        
        return chunks
    
    def generate_response_advanced(self, prompt: str, context: str = "", max_tokens: int = 512) -> str:
        """Generate response with Llama 2 chat format"""
        if not self.model_loaded or not self.model:
            return "Llama 2 7B model not loaded. Please initialize the model first."
        
        with self.generation_lock:
            try:
                if context:
                    formatted_prompt = f"""<s>[INST] <<SYS>>
You are a helpful Jira assistant. Answer questions about Jira issues based on the provided context. Be concise and accurate.
<</SYS>>

Context: {context}

Question: {prompt} [/INST]"""
                else:
                    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
                
                response = self.model(
                    formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    top_p=0.9,
                    top_k=40,
                    repeat_penalty=1.1,
                    stop=["</s>", "[INST]", "[/INST]"],
                    echo=False
                )
                
                result = response['choices'][0]['text'].strip()
                gc.collect()
                return result or "No response generated."
                
            except Exception as e:
                return f"Error: {str(e)}"
    
    def semantic_search_enhanced(self, query: str, top_k: int = 5) -> List[Dict]:
        """Enhanced search with optimized parameters - Fixed method name"""
        if not self.embedding_model or self.embeddings is None:
            return self.keyword_search(query, top_k)
        
        try:
            # Multi-stage search: exact match first, then semantic
            exact_matches = []
            for item in self.data:
                if query.lower() in item['issue_id'].lower():
                    exact_matches.append(item)
            
            if exact_matches:
                return exact_matches[:top_k]
            
            # Semantic search with optimized threshold
            query_embedding = self.embedding_model.encode([query])
            scores, indices = self.index.search(query_embedding.astype('float32'), min(top_k * 2, len(self.data)))
            
            threshold = self.search_params.get('similarity_threshold', 0.3)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.data) and scores[0][i] > threshold:
                    result = self.data[idx].copy()
                    result['similarity_score'] = float(scores[0][i])
                    results.append(result)
            
            return results[:top_k]
            
        except Exception as e:
            print(f"Enhanced search error: {e}")
            return self.keyword_search(query, top_k)
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Fallback keyword search"""
        if not self.data:
            return []
            
        query_words = query.lower().split()
        matches = []
        
        for item in self.data:
            text = f"{item.get('summary', '')} {item.get('description', '')} {item.get('issue_id', '')}".lower()
            score = sum(1 for word in query_words if word in text)
            if score > 0:
                item_copy = item.copy()
                item_copy['keyword_score'] = score
                matches.append(item_copy)
        
        matches.sort(key=lambda x: x['keyword_score'], reverse=True)
        return matches[:top_k]
    
    def handle_fast_queries(self, query: str) -> str:
        """Handle fast statistical or direct filtering queries"""
        if not self.data:
            return "No data available."

        query_lower = query.lower()

        # Handle bug count
        if 'count' in query_lower or 'how many' in query_lower:
            if 'bug' in query_lower:
                count = sum(c for t, c in self.stats_cache['type_counts'].items() if 'bug' in t.lower())
                return f"There are {count} bug issues in the dataset."
            elif 'total' in query_lower:
                return f"Total issues: {self.stats_cache['total_count']}"

        # Handle issue type distribution
        if 'type' in query_lower and 'distribution' in query_lower:
            types = self.stats_cache['type_counts'].most_common(5)
            result = "Issue type distribution:\n"
            for issue_type, count in types:
                result += f"• {issue_type}: {count}\n"
            return result.strip()

        if 'critical bug' in query_lower or 'show me critical bugs' in query_lower:
            filtered = [
                item for item in self.data
                if 'bug' in item['type'].lower() and 'critical' in item['priority'].lower()
            ]
            if not filtered:
                return "There are no critical bugs in the dataset."
            result = "Critical Bugs:\n"
            for item in filtered[:5]:  # Top 5 only
                summary = item['summary'] or item['description'][:100]
                result += f"- {item['issue_id']}: {summary}\n"
            return result.strip()

        return None
    
    def handle_complex_query_advanced(self, query: str) -> str:
        """Handle complex queries with full context"""
        relevant_items = self.semantic_search_enhanced(query, top_k=3)
        
        if not relevant_items:
            return "No relevant information found for your query."
        
        context_parts = []
        for item in relevant_items:
            context_parts.append(f"Issue {item['issue_id']} ({item['type']}): {item['summary']}")
            if item.get('description'):
                context_parts.append(f"Description: {item['description'][:200]}...")
        
        context = "\n".join(context_parts)
        return self.generate_response_advanced(query, context, max_tokens=self.max_length)
    
    def chat(self, message: str, history: List[List[str]]) -> Tuple[List[List[str]], str]:
        """Main chat function"""
        if not message.strip():
            return history, ""
        
        if not self.model_loaded and not self.data:
            response = "Please load training data first to start chatting."
        else:
            response = self.handle_fast_queries(message)
            if response is None:
                response = self.handle_complex_query_advanced(message)
        
        history.append([message, response])
        return history, ""
    
    def build_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """Build embeddings for semantic search"""
        if not self.embedding_model:
            return None
        
        texts = []
        for chunk in chunks:
            text = f"{chunk['issue_id']} {chunk['type']} {chunk['summary']} {chunk.get('description', '')[:200]}"
            texts.append(text)
        
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings

    def optimize_search_parameters(self) -> Dict[str, float]:
        """Optimize search parameters based on data characteristics"""
        if not self.data:
            return {
                'similarity_threshold': 0.3,
                'max_tokens': 1024,
                'top_k': 5
            }
        
        # Calculate optimal similarity threshold
        if self.embeddings is not None and len(self.data) > 10:
            # Sample queries and calculate average similarities
            sample_queries = [
                "show me bugs",
                "high priority issues", 
                "what issues are open",
                "critical problems"
            ]
            
            similarities = []
            for query in sample_queries:
                try:
                    query_embedding = self.embedding_model.encode([query])
                    scores, _ = self.index.search(query_embedding.astype('float32'), min(5, len(self.data)))
                    similarities.extend(scores[0])
                except:
                    continue
            
            if similarities:
                avg_sim = np.mean(similarities)
                optimal_threshold = max(0.2, avg_sim * 0.7)  # 70% of average similarity
            else:
                optimal_threshold = 0.3
        else:
            optimal_threshold = 0.3
        
        return {
            'similarity_threshold': optimal_threshold,
            'max_tokens': min(1024, len(self.data) * 2),  # Scale with dataset size
            'top_k': min(5, max(3, len(self.data) // 20))  # Adaptive top_k
        }


def create_interface():
    """Create Gradio interface"""
    chatbot = FineTunedJiraChatbot()
    
    def train_from_json(json_path, progress=gr.Progress()):
        if not json_path:
            return "Please select a JSON file first.", "Not trained", ""
        
        def progress_callback(msg):
            progress(0.5, desc=msg)
        
        success, pickle_path = chatbot.train_from_json(json_path, progress_callback)
        
        if success:
            status_msg = f"Training completed! {len(chatbot.data)} issues loaded."
            pickle_info = f"✅ Pickle file saved at:\n{pickle_path}"
            return status_msg, "Ready ✓", pickle_info
        else:
            return chatbot.training_status, "Failed ✗", ""
    
    def train_from_pickle(pickle_path, progress=gr.Progress()):
        if not pickle_path:
            return "Please select a pickle file first.", "Not trained", ""
        
        def progress_callback(msg):
            progress(0.5, desc=msg)
        
        success = chatbot.train_from_pickle(pickle_path, progress_callback)
        
        if success:
            return f"Data loaded! {len(chatbot.data)} issues.", "Ready ✓", ""
        else:
            return chatbot.training_status, "Failed ✗", ""
    
    def update_temperature(temp):
        chatbot.temperature = temp
        return f"Temperature: {temp}"
    
    with gr.Blocks(title="Llama 2 7B Jira Chatbot", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🦙 Llama 2 7B Jira Chatbot")
        gr.Markdown("*Train from JSON data, auto-convert to pickle format*")
        
        with gr.Tab("🔧 Training"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Train from JSON Data")
                    json_file = gr.File(label="Select JSON Training File", file_types=[".json"])
                    train_json_btn = gr.Button("Train from JSON", variant="primary", size="lg")
                    
                with gr.Column():
                    gr.Markdown("### Load Existing Pickle Data")
                    pickle_file = gr.File(label="Select Pickle Data File", file_types=[".pkl"])
                    train_pickle_btn = gr.Button("Load Pickle Data", variant="secondary")
            
            train_status = gr.Textbox(label="Training Status", interactive=False)
            model_status = gr.Textbox(label="Model Status", value="Not initialized", interactive=False)
            pickle_path_display = gr.Textbox(label="Generated Pickle File Path", interactive=False, visible=False)
            
            temperature_slider = gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="Temperature")
            temp_status = gr.Textbox(label="Current Temperature", value="0.7", interactive=False)
        
        with gr.Tab("💬 Chat"):
            gr.Markdown("## Ask Questions About Your Jira Data")
            
            chatbot_interface = gr.Chatbot(height=500, label="Chat History")
            
            with gr.Row():
                msg = gr.Textbox(placeholder="Ask about issues...", label="Question", scale=4)
                send_btn = gr.Button("Send", variant="primary", scale=1)
            
            clear_btn = gr.Button("Clear Chat", variant="secondary")
            
            gr.Examples([
                "How many total issues are there?",
                "What types of issues are most common?",
                "Show me critical bugs",
                "Analyze high priority issues",
                "What are the main problem areas?"
            ], inputs=msg)
        
        # Event handlers
        train_json_btn.click(train_from_json, inputs=[json_file], outputs=[train_status, model_status])
        train_pickle_btn.click(train_from_pickle, inputs=[pickle_file], outputs=[train_status, model_status])
        temperature_slider.change(update_temperature, inputs=[temperature_slider], outputs=[temp_status])
        
        msg.submit(chatbot.chat, [msg, chatbot_interface], [chatbot_interface, msg])
        send_btn.click(chatbot.chat, [msg, chatbot_interface], [chatbot_interface, msg])
        clear_btn.click(lambda: [], None, chatbot_interface)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, debug=True, server_name="0.0.0.0", server_port=7860)