class JIRAChatbot:
    def __init__(self, data_path, field_mapping=None):
        """Initialize JIRA chatbot
        
        Args:
            data_path (str): Path to the CSV file containing JIRA data
            field_mapping (dict): Optional mapping of custom field names to default names
        """
        self.field_mapping = field_mapping or {
            'Summary': 'Summary',
            'Description': 'Description',  # This might not exist in all datasets
            'Resolution': 'Resolution',
            'Status': 'Status',
            'Priority': 'Priority',
            'issue_key': 'Issue key',
            'Issue_Type': 'Issue Type',
            'Project_key': 'Project key',
            'Project_name': 'Project name'
        }
        
        try:
            self.df = pd.read_csv(data_path)
            # Rename columns if custom mapping provided
            if field_mapping:
                self.df = self.df.rename(columns=field_mapping)
                
            # Handle missing columns
            required_columns = ['Summary', 'Description', 'Resolution']
            for col in required_columns:
                if col not in self.df.columns:
                    self.df[col] = ''
                    
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

        self.text_processor = TextProcessor()
        self.intent_classifier = IntentClassifier()
        self.embedding_model = EmbeddingModel()
        self.issue_embeddings = None
        self.prepare_data()

    def prepare_data(self):
        """Prepare dataset with caching"""
        logger.info("0. prepare_data")
        
        try:
            # Create combined text field with better handling of missing values
            self.df['combined_text'] = (
                self.df[self.field_mapping['Summary']].fillna('') + ' ' +
                self.df[self.field_mapping['Description']].fillna('') + ' ' +
                self.df[self.field_mapping['Resolution']].fillna('') + ' ' +
                self.df[self.field_mapping['Issue_Type']].fillna('') + ' ' +
                self.df[self.field_mapping['Project_name']].fillna('')
            )
            logger.info("1. prepare_data")

            # Generate embeddings in batches
            batch_size = 32
            embeddings_list = []
            for i in range(0, len(self.df), batch_size):
                batch_texts = self.df['combined_text'].iloc[i:i+batch_size].tolist()
                batch_embeddings = self.embedding_model.get_sentence_embeddings(batch_texts)
                embeddings_list.append(batch_embeddings)
            
            self.issue_embeddings = np.vstack(embeddings_list)
            logger.info("2. prepare_data")

            # Extract NLP features with error handling
            self.df['nlp_features'] = self.df['combined_text'].apply(
                lambda x: self._safe_preprocess(x)
            )
            logger.info("3. prepare_data")
            
        except Exception as e:
            logger.error(f"Error in prepare_data: {str(e)}")
            raise

    def _safe_preprocess(self, text):
        """Safely preprocess text with error handling"""
        try:
            return self.text_processor.preprocess(text)
        except Exception as e:
            logger.warning(f"Error preprocessing text: {str(e)}")
            return {'entities': [], 'tokens': []}

    def get_analytics_response(self):
        """Generate analytics response"""
        try:
            stats = {
                'total_issues': len(self.df),
                'status_dist': self.df[self.field_mapping['Status']].value_counts(),
                'priority_dist': self.df[self.field_mapping['Priority']].value_counts(),
                'issue_type_dist': self.df[self.field_mapping['Issue_Type']].value_counts(),
                'project_dist': self.df[self.field_mapping['Project_name']].value_counts(),
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
                f"Issue Type Distribution:\n{stats['issue_type_dist']}\n"
                f"Project Distribution:\n{stats['project_dist']}\n"
                f"Average Sentiment: {stats['avg_sentiment']:.2f}"
            )
        except Exception as e:
            logger.error(f"Error generating analytics: {str(e)}")
            return "Sorry, I encountered an error while generating analytics."

    def get_response(self, query):
        """Generate response to query"""
        try:
            # Process query
            features = self.text_processor.preprocess(query)
            intent = self.intent_classifier.classify(features)

            # Handle analytics intent
            if intent in ['analytics', 'status', 'priority', 'type', 'project']:
                return self.get_analytics_response()

            # Find similar issues
            similar_issues = self.find_similar_issues(query)

            # Generate response
            response = "Found similar issues:\n\n"
            for _, issue in similar_issues.iterrows():
                response += (
                    f"Issue Key: {issue[self.field_mapping['issue_key']]}\n"
                    f"Type: {issue[self.field_mapping['Issue_Type']]}\n"
                    f"Summary: {issue[self.field_mapping['Summary']]}\n"
                    f"Status: {issue[self.field_mapping['Status']]}\n"
                    f"Priority: {issue[self.field_mapping['Priority']]}\n"
                    f"Project: {issue[self.field_mapping['Project_name']]}\n"
                )

                # Add sentiment
                sentiment = self.analyze_sentiment(issue['combined_text'])
                response += f"Sentiment: {sentiment:.2f}\n"

                # Add entities if available
                if isinstance(issue['nlp_features'], dict):
                    entities = issue['nlp_features'].get('entities', [])
                    if entities:
                        response += f"Entities: {', '.join([e[0] for e in entities])}\n"

                response += "-" * 50 + "\n"

            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "Sorry, I encountered an error while processing your query."