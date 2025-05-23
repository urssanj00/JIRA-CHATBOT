"""
#Environment nlp_chat
#Python 3.12.10
#pip install pandas faiss-cpu sentence-transformers transformers
Pre-compute Embeddings for NLP Chatbot
--------------------------------------
This script pre-processes input CSV data and saves embeddings to disk,
which drastically speeds up subsequent runs of your chatbot.
"""
import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import time
from tqdm import tqdm

# --- Step 1: Pre-compute and save text chunks and embeddings ---
def precompute_data(csv_path, output_dir):
    """
    Pre-processes the CSV file and saves text chunks and embeddings to disk.
    
    Args:
        csv_path: Path to your CSV file
        output_dir: Directory to save pre-computed data
    """
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file paths
    texts_path = os.path.join(output_dir, "text_chunks.pkl")
    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    metadata_path = os.path.join(output_dir, "metadata.pkl")
    
    print(f"Loading CSV data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Save metadata about the columns
    metadata = {
        "columns": list(df.columns),
        "row_count": len(df),
        "embedding_model": "all-MiniLM-L6-v2",
        "created_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_file": csv_path
    }
    
    # Convert rows to text with improved function
    def row_to_text(row):
        # Filter out empty or NaN values
        # Also limit the length of each value to avoid extremely long text
        MAX_VALUE_LENGTH = 500  # Maximum characters per field
        result = []
        for col in df.columns:
            if pd.notna(row[col]) and str(row[col]).strip():
                value = str(row[col]).strip()
                if len(value) > MAX_VALUE_LENGTH:
                    value = value[:MAX_VALUE_LENGTH] + "..."
                result.append(f"{col}: {value}")
        return " | ".join(result)
    
    print("Converting rows to text chunks...")
    texts = df.apply(row_to_text, axis=1).tolist()
    
    # Save texts to disk
    with open(texts_path, 'wb') as f:
        pickle.dump(texts, f)
    print(f"Saved {len(texts)} text chunks to {texts_path}")
    
    # Compute embeddings in batches
    print("Computing embeddings...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Process in batches to save memory
    BATCH_SIZE = 128
    embeddings_list = []
    
    for i in range(0, len(texts), BATCH_SIZE):
        end_idx = min(i + BATCH_SIZE, len(texts))
        batch_texts = texts[i:end_idx]
        print(f"Processing batch {i//BATCH_SIZE + 1}/{(len(texts) + BATCH_SIZE - 1)//BATCH_SIZE}")
        batch_embeddings = embedder.encode(batch_texts, show_progress_bar=True, 
                                          batch_size=32, convert_to_numpy=True)
        embeddings_list.append(batch_embeddings)
    
    embeddings = np.vstack(embeddings_list)
    
    # Save embeddings to disk
    np.save(embeddings_path, embeddings)
    print(f"Saved {len(embeddings)} embeddings to {embeddings_path}")
    
    # Save metadata to disk
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata to {metadata_path}")
    
    processing_time = time.time() - start_time
    print(f"Pre-processing completed in {processing_time:.2f} seconds")
    
    return {
        "texts_path": texts_path,
        "embeddings_path": embeddings_path,
        "metadata_path": metadata_path,
        "row_count": len(df),
        "processing_time": processing_time
    }


# --- Main Chatbot Implementation with Pre-computed Data ---
def chatbot_with_precomputed_data(embeddings_path, texts_path, metadata_path=None):
    """
    Runs the chatbot using pre-computed embeddings and text chunks.
    
    Args:
        embeddings_path: Path to pre-computed embeddings (.npy file)
        texts_path: Path to pre-computed text chunks (.pkl file)
        metadata_path: Optional path to metadata (.pkl file)
    """
    import os
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    import torch
    import faiss
    from transformers import pipeline, AutoTokenizer
    from sentence_transformers import SentenceTransformer
    from functools import lru_cache
    
    start_time = time.time()
    
    # Check if CUDA is available
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'CUDA' if device == 0 else 'CPU'}")
    
    # Load metadata if available
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        print(f"Loaded metadata: {len(metadata['columns'])} columns, {metadata['row_count']} rows")
    
    # Load text chunks
    print(f"Loading text chunks from {texts_path}...")
    with open(texts_path, 'rb') as f:
        texts = pickle.load(f)
    print(f"Loaded {len(texts)} text chunks")
    
    # Load embeddings
    print(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    print(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    
    # Create FAISS index
    print("Building FAISS index...")
    dim = embeddings.shape[1]
    
    # For smaller datasets (< 10k rows), use flat index
    if len(embeddings) < 10000:
        index = faiss.IndexFlatL2(dim)
    else:
        # For larger datasets, use IVF for faster search
        nlist = int(np.sqrt(len(embeddings)))
        quantizer = faiss.IndexFlatL2(dim)  
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        # Need to train the index
        print("Training IVF index...")
        index.train(embeddings)
    
    index.add(embeddings)
    print(f"Added {len(embeddings)} vectors to the index")
    
    # Load embedding model (only for queries, not for processing the whole dataset)
    print("Loading embedding model for queries...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    if device == 0:
        embedder = embedder.to(torch.device("cuda"))
    
    # Load FLAN-T5 model for answer generation
    print("Loading language model...")
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generator = pipeline("text2text-generation", model=model_name, tokenizer=tokenizer, device=device)
    
    # Cache queries to avoid re-encoding
    @lru_cache(maxsize=128)
    def encode_query(query):
        return embedder.encode([query])[0].reshape(1, -1)
    
    def retrieve_context(query, top_k=5):
        query_embedding = encode_query(query)
        # Convert to numpy array for FAISS
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        distances, indices = index.search(query_embedding, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # FAISS may return -1 for not found
                # Add score information to help with debugging
                score = distances[0][i]
                print(f"Match {i+1}: Score {score:.4f}")
                results.append((texts[idx], score))
        return [text for text, _ in results]
    
    def answer_question(question, top_k=5, max_tokens=512):
        context = retrieve_context(question, top_k)
        
        # Build prompt from top_k but trim if too long
        base_prompt = f"Answer the question based on the data:\n\n"
        suffix = f"\n\nQuestion: {question}"
        
        # First check if we can fit the entire context
        full_context = "\n".join(context)
        test_full_prompt = base_prompt + full_context + suffix
        tokenized = tokenizer(test_full_prompt, return_tensors="pt", truncation=False)
        full_token_count = len(tokenized["input_ids"][0])
        
        if full_token_count <= max_tokens:
            # We can use all the context
            final_prompt = test_full_prompt
        else:
            # Need to limit context - start with empty and add rows until limit
            prompt_body = ""
            for row in context:
                next_chunk = row + "\n"
                test_prompt = base_prompt + prompt_body + next_chunk + suffix
                # Don't use truncation when checking length to avoid misleading counts
                try:
                    tokenized = tokenizer(test_prompt, return_tensors="pt", truncation=False)
                    token_count = len(tokenized["input_ids"][0])
                    
                    if token_count < max_tokens:
                        prompt_body += next_chunk
                    else:
                        break
                except Exception as e:
                    # If we hit an error during tokenization (e.g., sequence too long), stop adding context
                    print(f"Warning: {str(e)}")
                    break
                    
            final_prompt = base_prompt + prompt_body + suffix
        
        # For the actual generation, use truncation to avoid errors
        # This ensures we stay within model limits even if our counting was off
        tokenized_for_generation = tokenizer(final_prompt, return_tensors="pt", truncation=True, max_length=max_tokens)
        token_count = len(tokenized_for_generation["input_ids"][0])
        print(f"Final prompt token count: {token_count}")
        
        # Use the truncated input_ids for generation
        response = generator(final_prompt, max_new_tokens=100, do_sample=False, truncation=True, max_length=max_tokens)
        return response[0]['generated_text']
    
    # --- Chat Loop with special commands ---
    print("\n✅ Ready! Ask a question based on your data (type 'exit' to quit):")
    print("Special commands:")
    print("  !info     - Show dataset information")
    print("  !sample   - Show a sample text chunk")
    print("  !topk N   - Change the number of context items (default: 5)")
    print("  !tokens N - Change the maximum tokens (default: 512)")
    
    # Initialize parameters that can be changed at runtime
    current_top_k = 5
    current_max_tokens = 512
    
    loading_time = time.time() - start_time
    print(f"Startup completed in {loading_time:.2f} seconds")
    
    try:
        while True:
            user_input = input("\n🗨️ You: ").strip()
            
            # Handle special commands
            if user_input.lower() in ['exit', 'quit']:
                break
            elif not user_input:
                print("Please enter a question.")
                continue
            elif user_input == "!info":
                if metadata_path and os.path.exists(metadata_path):
                    print("Dataset information:")
                    for key, value in metadata.items():
                        print(f"  - {key}: {value}")
                else:
                    print(f"Dataset info:")
                    print(f"  - Total chunks: {len(texts)}")
                    print(f"  - Embedding dimension: {dim}")
                print(f"  - Current settings: top_k={current_top_k}, max_tokens={current_max_tokens}")
                continue
            elif user_input == "!sample":
                print("Sample text chunk:")
                sample_idx = np.random.randint(0, len(texts))
                print(f"  {texts[sample_idx]}")
                continue
            elif user_input.startswith("!topk "):
                try:
                    new_top_k = int(user_input.split(" ")[1])
                    if new_top_k > 0:
                        current_top_k = new_top_k
                        print(f"Changed top_k to {current_top_k}")
                    else:
                        print("top_k must be positive")
                except:
                    print("Invalid format. Use !topk N where N is a positive integer")
                continue
            elif user_input.startswith("!tokens "):
                try:
                    new_max_tokens = int(user_input.split(" ")[1])
                    if 100 <= new_max_tokens <= 512:
                        current_max_tokens = new_max_tokens
                        print(f"Changed max_tokens to {current_max_tokens}")
                    else:
                        print("max_tokens must be between 100 and 512")
                except:
                    print("Invalid format. Use !tokens N where N is between 100 and 512")
                continue
                
            # Process regular questions
            try:
                print("Processing question...")
                question_start = time.time()
                answer = answer_question(user_input, top_k=current_top_k, max_tokens=current_max_tokens)
                question_time = time.time() - question_start
                print(f"🤖 Bot: {answer}")
                print(f"(Answered in {question_time:.2f} seconds)")
            except Exception as e:
                print(f"Error processing question: {str(e)}")
    except KeyboardInterrupt:
        print("\nExiting chat...")


if __name__ == "__main__":
    # Example usage:
    # Uncomment one of these sections to either pre-compute or run with pre-computed data
    
    # ------ Pre-compute data (run once) ------
    csv_path = r'C:\Sanjeev\VNIT_CLASSES\NLP_PROJ\dataset\GFG_FINAL.csv'
    output_dir = r'C:\Sanjeev\VNIT_CLASSES\NLP_PROJ\precomputed_data'
    
    #result = precompute_data(csv_path, output_dir)
    #print(f"Pre-computed data saved to {output_dir}")
    
    # ------ Run chatbot with pre-computed data (fast startup) ------
    # Uncomment this section after pre-computing data
    
    embeddings_path = r'C:\Sanjeev\VNIT_CLASSES\NLP_PROJ\precomputed_data\embeddings.npy'
    texts_path = r'C:\Sanjeev\VNIT_CLASSES\NLP_PROJ\precomputed_data\text_chunks.pkl'
    metadata_path = r'C:\Sanjeev\VNIT_CLASSES\NLP_PROJ\precomputed_data\metadata.pkl'
    
    chatbot_with_precomputed_data(embeddings_path, texts_path, metadata_path)