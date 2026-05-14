from util import *

# Add your import statements here
import re
import nltk
import spacy
from nltk.tokenize import sent_tokenize


class SentenceSegmentation():

	def __init__(self):
		# Load spaCy model (students may use this if needed)
		self.nlp = spacy.load("en_core_web_sm")

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None

		# Top-down rule: split on '.', '!', or '?' when followed by
		# whitespace and an uppercase letter (marks a likely new sentence).
		# Known limitation: fails on abbreviations like Dr., U.S.A., decimals, etc.
		segmentedText = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
		# Remove any empty strings produced by the split
		segmentedText = [s.strip() for s in segmentedText if s.strip()]

		return segmentedText


	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None

		# NLTK Punkt uses an unsupervised ML algorithm (Kiss & Strunk, 2006).
		# It learns abbreviation types and sentence starters from the corpus,
		# making it robust to abbreviations that break the naive approach.
		segmentedText = sent_tokenize(text)

		return segmentedText


	def spacySegmenter(self, text):
		"""
		Sentence Segmentation using spaCy

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None

		# spaCy uses a trained dependency parser to detect sentence boundaries
		# by understanding syntactic structure, achieving the highest accuracy.
		doc = self.nlp(text)
		segmentedText = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

		return segmentedText
