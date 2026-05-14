from util import *

# Add your import statements here
import math
from collections import Counter
import time




class InformationRetrieval():

	def __init__(self):
		self.index = None

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""

		index = None

		#Fill in code here
		start_time = time.time()

		# Store document IDs and total number of documents
		self.docIDs = docIDs
		N = len(docs)

		# Flatten each document (list of sentences of tokens) into a single token list
		# and compute term frequency for each document
		self.docTermFreqs = {}
		self.docLengths = {}  # Store document vector magnitudes for cosine normalization

		# Inverted index: term -> {docID: tf}
		index = {}

		for i, doc in enumerate(docs):
			docID = docIDs[i]
			# Flatten: doc is a list of sentences, each sentence is a list of tokens
			tokens = []
			for sentence in doc:
				tokens.extend([token.lower() for token in sentence])

			# Compute term frequency
			tf = Counter(tokens)
			self.docTermFreqs[docID] = tf

			# Build inverted index
			for term in tf:
				if term not in index:
					index[term] = {}
				index[term][docID] = tf[term]

		# Compute IDF for each term
		self.idf = {}
		for term, postings in index.items():
			df = len(postings)  # number of documents containing the term
			self.idf[term] = math.log10(N / df)

		# Pre-compute TF-IDF document vectors and their magnitudes
		self.docTFIDF = {}
		for docID in docIDs:
			tfidf = {}
			tf = self.docTermFreqs[docID]
			for term, freq in tf.items():
				# Use log-weighted TF: 1 + log10(tf)
				tf_weight = 1 + math.log10(freq) if freq > 0 else 0
				tfidf[term] = tf_weight * self.idf.get(term, 0)
			self.docTFIDF[docID] = tfidf

			# Compute magnitude of document vector
			magnitude = math.sqrt(sum(w * w for w in tfidf.values()))
			self.docLengths[docID] = magnitude

		self.index = index
		end_time = time.time()
		print(f"\nTime taken to build index: {end_time - start_time:.4f} seconds")


	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []

		#Fill in code here
		start_time = time.time()

		for query in queries:
			# Flatten query: list of sentences of tokens -> single token list
			queryTokens = []
			for sentence in query:
				queryTokens.extend([token.lower() for token in sentence])

			# Compute query TF-IDF vector
			queryTF = Counter(queryTokens)
			queryTFIDF = {}
			for term, freq in queryTF.items():
				tf_weight = 1 + math.log10(freq) if freq > 0 else 0
				queryTFIDF[term] = tf_weight * self.idf.get(term, 0)

			# Compute query vector magnitude
			queryMagnitude = math.sqrt(sum(w * w for w in queryTFIDF.values()))

			# Compute cosine similarity with each document
			scores = []
			for docID in self.docIDs:
				dotProduct = 0
				docVector = self.docTFIDF[docID]
				# Only iterate over query terms (sparse dot product)
				for term, queryWeight in queryTFIDF.items():
					if term in docVector:
						dotProduct += queryWeight * docVector[term]

				# Cosine similarity = dot / (|q| * |d|)
				denom = queryMagnitude * self.docLengths[docID]
				if denom > 0:
					cosineSim = dotProduct / denom
				else:
					cosineSim = 0

				scores.append((docID, cosineSim))

			# Sort by score descending
			scores.sort(key=lambda x: x[1], reverse=True)
			doc_IDs_ordered.append([docID for docID, _ in scores])
	
		end_time = time.time()
		print(f"Time taken to rank all queries: {end_time - start_time:.4f} seconds")

		return doc_IDs_ordered




