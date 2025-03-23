# src/processor/intent_classifier.py
class IntentClassifier:
    def __init__(self):
        """Initialize intent classification"""
        self.intents = {
            'search': ['find', 'search', 'look', 'similar'],
            'analytics': ['analyze', 'statistics', 'distribution', 'trend'],
            'status': ['status', 'progress', 'state'],
            'priority': ['priority', 'urgent', 'importance'],
            'resolution': ['resolution', 'solved', 'fixed']
        }

    def classify(self, text_features):
        """Classify user intent"""
        tokens = set(text_features['tokens'])

        intent_scores = {
            intent: len(tokens.intersection(keywords))
            for intent, keywords in self.intents.items()
        }

        return max(intent_scores.items(), key=lambda x: x[1])[0]