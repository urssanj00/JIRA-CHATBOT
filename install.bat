#!/bin/bash
# install.sh

# Create and activate conda environment
conda create -n jira_chatbot_env python=3.9 -y
conda activate jira_chatbot_env

# Install requirements
pip install -r requirements.txt

# Download spacy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download(['punkt', 'stopwords', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'])"

# Install the package in development mode
pip install -e .

echo "Installation complete! Activate the environment with: conda activate jira_chatbot_env"