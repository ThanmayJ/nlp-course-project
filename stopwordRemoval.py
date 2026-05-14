from util import *

# Add your import statements here
import nltk
from nltk.corpus import stopwords

# Ensure stopwords corpus is available
try:
	nltk.download('stopwords', quiet=True)
except Exception:
	pass


class StopwordRemoval():

	def __init__(self):
		# Load NLTK's curated English stopword list once at initialisation.
		# This set contains ~179 common English function words (articles,
		# prepositions, conjunctions, pronouns, auxiliary verbs) that carry
		# little discriminative information for information retrieval.
		self._stopwords = set(stopwords.words('english'))

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stopwordRemovedText = None

		# Filter out any token whose lowercase form appears in the stopword set.
		# Case-insensitive comparison ensures "The" and "the" are both removed.
		# Tokens not in the stopword list (content words) are kept unchanged.
		stopwordRemovedText = []
		for sentence in text:
			filtered_sentence = [
				token for token in sentence
				if token.lower() not in self._stopwords
			]
			stopwordRemovedText.append(filtered_sentence)

		return stopwordRemovedText


