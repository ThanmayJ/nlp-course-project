from util import *

# Add your import statements here
import re
import nltk
import spacy
from nltk.tokenize import TreebankWordTokenizer


class Tokenization():

	def __init__(self):
		# Load spaCy model once so spacyTokenizer can reuse it
		self.nlp = spacy.load("en_core_web_sm")
		# Instantiate Penn Treebank tokenizer once for efficiency
		self._ptb_tokenizer = TreebankWordTokenizer()

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = None

		# Top-down approach: extract word characters and apostrophes using regex.
		# Handles basic words and contractions (e.g., "don't" kept as one token).
		# Known failures: misses possessive splits, hyphenated compounds, currency.
		tokenizedText = []
		for sentence in text:
			# \b\w+(?:'\w+)?\b captures plain words and words with apostrophes
			tokens = re.findall(r"\b\w+(?:'\w+)?\b", sentence)
			tokenizedText.append(tokens)

		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = None

		# Penn Treebank tokenizer uses hand-crafted regex rules derived from
		# corpus conventions (hybrid top-down + bottom-up). It correctly splits
		# contractions ("don't" -> ["do", "n't"]), possessives ("John's" -> ["John", "'s"]),
		# and isolates punctuation.
		tokenizedText = []
		for sentence in text:
			tokens = self._ptb_tokenizer.tokenize(sentence)
			tokenizedText.append(tokens)

		return tokenizedText



	def spacyTokenizer(self, text):
		"""
		Tokenization using spaCy

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = None

		# spaCy applies prefix/suffix/infix rules (top-down) combined with
		# a learned exception dictionary (bottom-up), making it a hybrid tokenizer.
		# It handles contractions, URLs, emails, and multi-word expressions.
		tokenizedText = []
		for sentence in text:
			doc = self.nlp(sentence)
			tokens = [token.text for token in doc]
			tokenizedText.append(tokens)

		return tokenizedText
