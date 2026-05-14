from util import *

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import time

class ImprovedInformationRetrieval():
	"""
	Improved IR System using Latent Semantic Analysis (LSA).
	LSA helps to solve the term-mismatch and synonymy problems inherent in standard
	Vector Space Models by projecting documents and queries into a lower-dimensional
	semantic subspace.
	"""

	def __init__(self, n_components=100):
		self.n_components = n_components
		self.vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False, preprocessor=lambda x: x)
		self.svd = TruncatedSVD(n_components=n_components, random_state=42)
		self.docIDs = None
		self.doc_embeddings = None

	def buildIndex(self, docs, docIDs):
		"""
		Trains LSA model on the corpus and computes low-dimensional semantic vectors
		for all documents.
		"""
		print(f"[LSA] Building Index with {self.n_components} latent dimensions...")
		start_time = time.time()
		
		self.docIDs = docIDs

		# Flatten document structures for input to TfidfVectorizer
		# Input format expected: list of lists of strings
		flattened_docs = []
		for doc in docs:
			# Flatten list of sentences into a single list of lowercase tokens
			all_tokens = []
			for sentence in doc:
				all_tokens.extend([str(tok).lower() for tok in sentence])
			flattened_docs.append(all_tokens)

		# 1. Build TF-IDF Matrix
		tfidf_matrix = self.vectorizer.fit_transform(flattened_docs)

		# 2. Apply Truncated SVD (LSA)
		# Dynamically reduce components if corpus dimension is smaller
		actual_components = min(self.n_components, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0])
		if actual_components != self.n_components:
			print(f"[LSA] Adjusting components to {actual_components} based on dimension limits.")
			self.svd = TruncatedSVD(n_components=actual_components, random_state=42)

		self.doc_embeddings = self.svd.fit_transform(tfidf_matrix)

		end_time = time.time()
		print(f"Time taken to build LSA index: {end_time - start_time:.4f} seconds")
		print(f"[LSA] Total Explained Variance: {np.sum(self.svd.explained_variance_ratio_):.4f}")


	def rank(self, queries):
		"""
		Projects queries into the learned LSA semantic subspace and ranks
		documents based on cosine similarity.
		"""
		print("[LSA] Ranking queries...")
		start_time = time.time()
		
		# Flatten queries
		flattened_queries = []
		for query in queries:
			q_tokens = []
			for sentence in query:
				q_tokens.extend([str(tok).lower() for tok in sentence])
			flattened_queries.append(q_tokens)

		# 1. Transform queries to TF-IDF vectorizer vocabulary space
		q_tfidf = self.vectorizer.transform(flattened_queries)

		# 2. Project queries into SVD subspace
		q_embeddings = self.svd.transform(q_tfidf)

		# 3. Batch Compute Cosine Similarities
		similarities = cosine_similarity(q_embeddings, self.doc_embeddings)

		# 4. Sort document IDs based on scores per query
		doc_IDs_ordered = []
		for query_sims in similarities:
			# Argsort gives ascending indices, we reverse it
			ranked_indices = np.argsort(query_sims)[::-1]
			sorted_ids = [self.docIDs[idx] for idx in ranked_indices]
			doc_IDs_ordered.append(sorted_ids)

		end_time = time.time()
		print(f"Time taken to rank all queries (LSA): {end_time - start_time:.4f} seconds")
		
		return doc_IDs_ordered
