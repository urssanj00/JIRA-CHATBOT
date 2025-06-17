# src/models/embeddings.py
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
import torch
import logging

logger = logging.getLogger(__name__)


class EmbeddingModel:
    def __init__(self):
        """Initialize embedding models"""
        try:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            self.bert_tokenizer = BertTokenizer.from_pretrained(
                'bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        except Exception as e:
            logger.error(f"Error initializing embedding models: {str(e)}")
            raise

    def get_sentence_embeddings(self, text):
        """Get sentence transformer embeddings

        Args:
            text (str or list): Single text or list of texts to embed

        Returns:
            numpy.ndarray: Embeddings array
        """
        try:
            if isinstance(text, str):
                text = [text]
            return self.sentence_transformer.encode(text)
        except Exception as e:
            logger.error(f"Error getting sentence embeddings: {str(e)}")
            raise

    def get_bert_embeddings(self, text):
        """Get BERT embeddings

        Args:
            text (str): Text to embed

        Returns:
            torch.Tensor: BERT embeddings
        """
        try:
            inputs = self.bert_tokenizer(text, return_tensors="pt",
                                         padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)
        except Exception as e:
            logger.error(f"Error getting BERT embeddings: {str(e)}")
            raise
