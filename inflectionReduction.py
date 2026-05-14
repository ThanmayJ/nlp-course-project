from util import *

# Add your import statements here
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Ensure required NLTK resources are available
for resource in ['wordnet', 'averaged_perceptron_tagger']:
	try:
		nltk.download(resource, quiet=True)
	except Exception:
		pass


def _get_wordnet_pos(treebank_tag):
	"""
	Map a Penn Treebank POS tag to the corresponding WordNet POS constant.
	Defaults to NOUN if the tag is unrecognised.
	"""
	if treebank_tag.startswith('J'):
		return wordnet.ADJ
	elif treebank_tag.startswith('V'):
		return wordnet.VERB
	elif treebank_tag.startswith('R'):
		return wordnet.ADV
	else:
		# Default to NOUN for everything else (NN, NNS, NNP, etc.)
		return wordnet.NOUN


class InflectionReduction:

	def porterStemmer(self, text):
		"""
		Inflection Reduction using Porter Stemmer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed tokens representing a sentence
		"""

		reducedText = None

		# Porter Stemmer strips suffixes using rule-based heuristics.
		# Very fast but may produce non-words (e.g., "studies" -> "studi").
		# Risk of over-stemming: unrelated words can collapse to the same stem
		# (e.g., "universal" and "university" both -> "univers").
		stemmer = PorterStemmer()
		reducedText = []
		for sentence in text:
			stemmed_sentence = [stemmer.stem(token) for token in sentence]
			reducedText.append(stemmed_sentence)

		return reducedText



	def wordnetLemmatizer(self, text):
		"""
		Inflection Reduction using WordNet Lemmatizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			lemmatized tokens representing a sentence
		"""

		reducedText = None

		# WordNet Lemmatizer looks up the canonical base form in the WordNet
		# lexical database. Using POS tags gives better accuracy:
		# e.g., "better" (ADJ) -> "good"; "was" (VERB) -> "be".
		# Always returns a valid English word (unlike stemming).
		lemmatizer = WordNetLemmatizer()
		reducedText = []
		for sentence in text:
			# POS-tag the whole sentence for context-aware lemmatization
			pos_tags = nltk.pos_tag(sentence)
			lemmatized_sentence = [
				lemmatizer.lemmatize(token, _get_wordnet_pos(pos))
				for token, pos in pos_tags
			]
			reducedText.append(lemmatized_sentence)

		return reducedText



	def reduce(self, text):
		"""
		Wrapper function for inflection reduction.
		Students may choose which method to call
		or extend this function to support both options.
		"""

		reducedText = None

		# Using the WordNet Lemmatizer as the default reduction method because:
		# - It always produces valid English words (important for readability).
		# - POS-aware lemmatization preserves semantic meaning better than stemming.
		# - Avoids over-stemming artefacts (e.g., "univers" for both "universal"
		#   and "university") that would incorrectly merge unrelated terms.
		reducedText = self.wordnetLemmatizer(text)

		return reducedText
