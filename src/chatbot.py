# src/chatbot.py
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
from .processor.text_processor import TextProcessor
from .processor.intent_classifier import IntentClassifier
from .models.embeddings import EmbeddingModel


class JIRAChatbot:
    def __init__(self, data_path):
        """Initialize JIRA chatbot"""
        self.df = pd.read_csv(data_path)
        self.text_processor = TextProcessor()
        self.intent_classifier = IntentClassifier()
        self.embedding_model = EmbeddingModel()
        self.prepare_data()

    def prepare_data(self):
        """Prepare dataset"""
        self.df['combined_text'] = (
                self.df['summary'].fillna('') + ' ' +
                self.df['description'].fillna('') + ' ' +
                self.df['resolution'].fillna('')
        )

        # Generate embeddings
        self.issue_embeddings = np.vstack([
            self.embedding_model.get_sentence_embeddings(text)
            for text in self.df['combined_text']
        ])

        # Extract NLP features
        self.df['nlp_features'] = self.df['combined_text'].apply(
            self.text_processor.preprocess
        )

    def analyze_sentiment(self, text):
        """Analyze text sentiment"""
        return TextBlob(text).sentiment.polarity

    def find_similar_issues(self, query, n=5):
        """Find similar issues"""
        query_embedding = self.embedding_model.get_sentence_embeddings(query)
        similarities = cosine_similarity(
            [query_embedding],
            self.issue_embeddings
        )[0]

        return self.df.iloc[similarities.argsort()[-n:][::-1]]

    def get_analytics_response(self):
        """Generate analytics response"""
        stats = {
            'total_issues': len(self.df),
            'status_dist': self.df['status'].value_counts(),
            'priority_dist': self.df['priority'].value_counts(),
            'avg_sentiment': np.mean([
                self.analyze_sentiment(text)
                for text in self.df['combined_text']
            ])
        }

        return (
            f"Analytics Summary:\n"
            f"Total Issues: {stats['total_issues']}\n"
            f"Status Distribution:\n{stats['status_dist']}\n"
            f"Priority Distribution:\n{stats['priority_dist']}\n"
            f"Average Sentiment: {stats['avg_sentiment']:.2f}"
        )

    def get_response(self, query):
        """Generate response to query"""
        # Process query
        features = self.text_processor.preprocess(query)
        intent = self.intent_classifier.classify(features)

        # Handle analytics intent
        if intent in ['analytics', 'status', 'priority']:
            return self.get_analytics_response()

        # Find similar issues
        similar_issues = self.find_similar_issues(query)

        # Generate response
        response = "Found similar issues:\n\n"
        for _, issue in similar_issues.iterrows():
            response += (
                f"Issue Key: {issue['issue_key']}\n"
                f"Summary: {issue['summary']}\n"
                f"Status: {issue['status']}\n"
                f"Priority: {issue['priority']}\n"
            )

            # Add sentiment
            sentiment = self.analyze_sentiment(issue['combined_text'])
            response += f"Sentiment: {sentiment:.2f}\n"

            # Add entities
            if isinstance(issue['nlp_features'], dict):
                entities = issue['nlp_features']['entities']
                if entities:
                    response += f"Entities: {', '.join([e[0] for e in entities])}\n"

            response += "-" * 50 + "\n"

        return response