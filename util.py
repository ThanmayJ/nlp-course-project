# Add your import statements here
import json
import os
import nltk
import spacy


# Add any utility functions here

# Download required NLTK resources
for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    try:
        nltk.download(resource, quiet=True)
    except Exception as e:
        print(f"Warning: Could not download {resource}: {e}")

# Load spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def load_json(path):
	"""Load a JSON file and return its contents."""
	with open(path, 'r') as f:
		return json.load(f)
