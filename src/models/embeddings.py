# src/models/embeddings.py
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
import torch


class EmbeddingModel:
    def __init__(self):
        """Initialize embedding models"""
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

    def get_sentence_embeddings(self, text):
        """Get sentence transformer embeddings"""
        return self.sentence_transformer.encode([text])[0]

    def get_bert_embeddings(self, text):
        """Get BERT embeddings"""
        inputs = self.bert_tokenizer(text, return_tensors="pt",
                                     padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)