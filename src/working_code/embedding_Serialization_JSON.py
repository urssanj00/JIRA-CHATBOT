import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from typing import Dict, List, Tuple, Any
import re
from datetime import datetime

class JiraTableEmbeddingSerializer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding serializer for Jira table data
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.encoder = SentenceTransformer(model_name)
        self.field_weights = {
            'summary': 0.3,
            'description': 0.25,
            'issue_type': 0.1,
            'status': 0.1,
            'priority': 0.1,
            'component': 0.05,
            'assignee': 0.05,
            'reporter': 0.05
        }
        
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text data"""
        if pd.isna(text) or text == '':
            return ""
        
        # Remove URLs, image references, and special characters
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', str(text))
        text = re.sub(r'!\S+\.\S+!', '', text)  # Remove image references
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def create_field_embeddings(self, row: pd.Series) -> Dict[str, np.ndarray]:
        """Create embeddings for individual fields"""
        field_embeddings = {}
        
        # Key fields for embedding
        fields_to_embed = {
            'summary': row.get('Summary', ''),
            'description': row.get('Description', ''),
            'issue_type': row.get('Issue Type', ''),
            'status': row.get('Status', ''),
            'priority': row.get('Priority', ''),
            'component': row.get('Component/s', ''),
            'labels': row.get('Labels', ''),
            'environment': row.get('Environment', ''),
            'comments': row.get('Comment', '')
        }
        
        for field, value in fields_to_embed.items():
            cleaned_text = self.preprocess_text(value)
            if cleaned_text:
                embedding = self.encoder.encode(cleaned_text, normalize_embeddings=True)
                field_embeddings[field] = embedding
            else:
                # Zero embedding for empty fields
                field_embeddings[field] = np.zeros(self.encoder.get_sentence_embedding_dimension())
                
        return field_embeddings
    
    def create_composite_embedding(self, field_embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Create weighted composite embedding from field embeddings"""
        composite = np.zeros(self.encoder.get_sentence_embedding_dimension())
        total_weight = 0
        
        for field, embedding in field_embeddings.items():
            weight = self.field_weights.get(field, 0.05)  # Default small weight
            composite += weight * embedding
            total_weight += weight
            
        # Normalize by total weight
        if total_weight > 0:
            composite = composite / total_weight
            
        return composite
    
    def create_contextual_text(self, row: pd.Series) -> str:
        """Create structured text representation for contextual embedding"""
        parts = []
        
        # Priority information
        if pd.notna(row.get('Priority')):
            parts.append(f"Priority: {row['Priority']}")
            
        # Issue metadata
        if pd.notna(row.get('Issue Type')):
            parts.append(f"Type: {row['Issue Type']}")
            
        if pd.notna(row.get('Status')):
            parts.append(f"Status: {row['Status']}")
            
        # Main content
        if pd.notna(row.get('Summary')):
            parts.append(f"Summary: {self.preprocess_text(row['Summary'])}")
            
        if pd.notna(row.get('Description')):
            desc = self.preprocess_text(row['Description'])
            if len(desc) > 500:  # Truncate very long descriptions
                desc = desc[:500] + "..."
            parts.append(f"Description: {desc}")
            
        # Additional context
        if pd.notna(row.get('Component/s')):
            parts.append(f"Component: {row['Component/s']}")
            
        if pd.notna(row.get('Assignee')):
            parts.append(f"Assignee: {row['Assignee']}")
            
        return " | ".join(parts)
    
    def serialize_row(self, row: pd.Series) -> Dict[str, Any]:
        """Serialize a single row into embedding-based format"""
        # Create field-specific embeddings
        field_embeddings = self.create_field_embeddings(row)
        
        # Create composite embedding
        composite_embedding = self.create_composite_embedding(field_embeddings)
        
        # Create contextual text embedding
        contextual_text = self.create_contextual_text(row)
        contextual_embedding = self.encoder.encode(contextual_text, normalize_embeddings=True)
        
        # Metadata for training
        metadata = {
            'issue_key': row.get('Issue key', ''),
            'issue_type': row.get('Issue Type', ''),
            'status': row.get('Status', ''),
            'priority': row.get('Priority', ''),
            'created': row.get('Created', ''),
            'updated': row.get('Updated', ''),
            'project': row.get('Project name', ''),
            'severity': row.get('Symptom Severity', ''),
            'original_text_length': len(str(row.get('Description', '')))
        }
        
        return {
            'field_embeddings': {k: v.tolist() for k, v in field_embeddings.items()},
            'composite_embedding': composite_embedding.tolist(),
            'contextual_embedding': contextual_embedding.tolist(),
            'contextual_text': contextual_text,
            'metadata': metadata,
            'raw_summary': self.preprocess_text(row.get('Summary', '')),
            'raw_description': self.preprocess_text(row.get('Description', ''))[:1000]  # Truncate for training
        }
    
    def generate_qa_pairs(self, serialized_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate question-answer pairs for fine-tuning"""
        qa_pairs = []
        metadata = serialized_data['metadata']
        
        # Template-based QA generation
        qa_templates = [
            {
                'question': f"What is the status of issue {metadata['issue_key']}?",
                'answer': f"The status of issue {metadata['issue_key']} is {metadata['status']}.",
                'type': 'status_lookup'
            },
            {
                'question': f"What is the priority of {metadata['issue_key']}?",
                'answer': f"Issue {metadata['issue_key']} has {metadata['priority']} priority.",
                'type': 'priority_lookup'
            },
            {
                'question': f"Can you summarize issue {metadata['issue_key']}?",
                'answer': f"Issue {metadata['issue_key']}: {serialized_data['raw_summary']}",
                'type': 'summarization'
            },
            {
                'question': f"What type of issue is {metadata['issue_key']}?",
                'answer': f"{metadata['issue_key']} is a {metadata['issue_type']}.",
                'type': 'classification'
            }
        ]
        
        # Add contextual questions
        if serialized_data['raw_description']:
            qa_templates.append({
                'question': f"Can you provide details about {metadata['issue_key']}?",
                'answer': f"Here are the details for {metadata['issue_key']}: {serialized_data['raw_description']}",
                'type': 'detailed_lookup'
            })
            
        # Add comparison/filtering questions
        qa_templates.extend([
            {
                'question': f"Show me {metadata['priority'].lower()} priority issues",
                'answer': f"Here's a {metadata['priority'].lower()} priority issue: {metadata['issue_key']} - {serialized_data['raw_summary']}",
                'type': 'filtering'
            },
            {
                'question': f"List {metadata['issue_type'].lower()} issues",
                'answer': f"Here's a {metadata['issue_type'].lower()}: {metadata['issue_key']} - {serialized_data['raw_summary']}",
                'type': 'filtering'
            }
        ])
        
        # Attach embeddings to each QA pair
        for qa in qa_templates:
            qa_with_embeddings = qa.copy()
            qa_with_embeddings.update({
                'question_embedding': self.encoder.encode(qa['question'], normalize_embeddings=True).tolist(),
                'context_embedding': serialized_data['contextual_embedding'],
                'composite_embedding': serialized_data['composite_embedding'],
                'metadata': metadata
            })
            qa_pairs.append(qa_with_embeddings)
            
        return qa_pairs
    
    def process_dataframe(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """Process entire dataframe and generate training data"""
        serialized_data = []
        all_qa_pairs = []
        
        for idx, row in df.iterrows():
            try:
                # Serialize row
                row_data = self.serialize_row(row)
                serialized_data.append(row_data)
                
                # Generate QA pairs
                qa_pairs = self.generate_qa_pairs(row_data)
                all_qa_pairs.extend(qa_pairs)
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
                
        return serialized_data, all_qa_pairs
    
    def create_training_format(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Convert to standard fine-tuning format"""
        training_data = []
        
        for qa in qa_pairs:
            # Standard instruction-following format
            training_example = {
                'instruction': "You are a helpful assistant that answers questions about Jira issues and bugs.",
                'input': qa['question'],
                'output': qa['answer'],
                'question_embedding': qa['question_embedding'],
                'context_embedding': qa['context_embedding'],
                'metadata': qa['metadata'],
                'task_type': qa['type']
            }
            training_data.append(training_example)
            
        return training_data

# Example usage and processing
def process_jira_data():
    """Process the provided Jira data"""
    
    # Parse the provided data (assuming CSV format)
    data = {
        'Summary': [
            "Sourcetree repository tab width automatically adjusted with repository name is bad with short names",
            "Stashes don't show untracked files when clicking on them",
            "Make the list of tags more usable (e.g. descending sort)",
            "GlucoTru Reviews"  # This appears to be spam
        ],
        'Issue key': ['SRCTREEWIN-14221', 'SRCTREEWIN-14215', 'SRCTREEWIN-14198', 'SRCTREEWIN-14167'],
        'Issue Type': ['Bug', 'Bug', 'Suggestion', 'Suggestion'],
        'Status': ['Short Term Backlog', 'Short Term Backlog', 'Under Consideration', 'Closed'],
        'Priority': ['Low', 'Low', '', ''],
        'Description': [
            "SourceTree 3.4.13 Change: Sourcetree repository tab width should automatically adjust with repository name If the project name is short, the X for to close the Tab overwrite the tab name so you close the tab instead of select it.",
            "I could swear that in the past if I created a stash and had it include untracked files, when I clicked on the stash and it displayed the files modified it would also display untracked files. It doesn't do that now though.",
            "The sort order of tags in the repo sidebar has the most recently added (highest) tags at the bottom of the list. IMHO the most recent tags are the most likely to be used to refer to specific changesets.",
            "Spam content about GlucoTru supplement"
        ],
        'Component/s': ['General', 'Stash', 'UX', 'hg'],
        'Project name': ['Sourcetree for Windows'] * 4
    }
    
    df = pd.DataFrame(data)
    
    # Initialize serializer
    serializer = JiraTableEmbeddingSerializer()
    
    # Process data
    serialized_data, qa_pairs = serializer.process_dataframe(df)
    
    # Create training format
    training_data = serializer.create_training_format(qa_pairs)
    
    return serialized_data, training_data

# Demo execution
if __name__ == "__main__":
    serialized_data, training_data = process_jira_data()
    
    print("=== Embedding-based Serialization Results ===")
    print(f"Processed {len(serialized_data)} issues")
    print(f"Generated {len(training_data)} training examples")
    
    # Show sample serialized data structure
    print("\n=== Sample Serialized Data Structure ===")
    sample = serialized_data[0]
    print("Keys:", list(sample.keys()))
    print("Embedding dimensions:")
    for key, value in sample['field_embeddings'].items():
        print(f"  {key}: {len(value)} dimensions")
    
    print(f"Composite embedding: {len(sample['composite_embedding'])} dimensions")
    print(f"Contextual embedding: {len(sample['contextual_embedding'])} dimensions")
    
    # Show sample training data
    print("\n=== Sample Training Example ===")
    sample_training = training_data[0]
    print("Instruction:", sample_training['instruction'])
    print("Input:", sample_training['input'])
    print("Output:", sample_training['output'])
    print("Task type:", sample_training['task_type'])
    print("Question embedding shape:", len(sample_training['question_embedding']))
    
    # Save results
    with open('jira_embeddings_training_data.json', 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"\nTraining data saved to 'jira_embeddings_training_data.json'")
    print("Ready for LLM fine-tuning!")