# src/processor/text_processor.py
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re


class TextProcessor:
    def __init__(self):
        """Initialize text processing components"""
        self.nlp = spacy.load('en_core_web_sm')
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, text):
        """Preprocess text and extract features"""
        if not isinstance(text, str):
            return None

        # Basic cleaning
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)

        # spaCy processing
        doc = self.nlp(text)

        return {
            'tokens': [token.text for token in doc],
            'lemmas': [token.lemma_ for token in doc],
            'pos_tags': [token.pos_ for token in doc],
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'noun_chunks': [chunk.text for chunk in doc.noun_chunks],
            'sentences': sent_tokenize(text)
        }