
#1. First, create and set up the Conda environment:

#```bash
# Create new conda environment
conda create -n jira_chatbot_env python=3.9
conda activate jira_chatbot_env

# Install required packages
conda install -c conda-forge pandas numpy spacy transformers nltk scikit-learn
conda install -c pytorch pytorch
conda install -c huggingface sentence-transformers
conda install -c conda-forge textblob
python -m spacy download en_core_web_sm
#```
