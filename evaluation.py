from util import *

# Add your import statements here
import math


class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1

		#Fill in code here
		top_k = query_doc_IDs_ordered[:k]
		true_set = set(true_doc_IDs)
		relevant_count = sum(1 for doc_id in top_k if doc_id in true_set)
		precision = relevant_count / k

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""
		meanPrecision = -1

		#Fill in code here
		# Build a mapping from query_num to list of relevant doc IDs
		relevance = {}
		for qrel in qrels:
			query_num = int(qrel["query_num"])
			doc_id = int(qrel["id"])
			if query_num not in relevance:
				relevance[query_num] = []
			relevance[query_num].append(doc_id)

		precisions = []
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = relevance.get(query_id, [])
			precision = self.queryPrecision(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			precisions.append(precision)

		meanPrecision = sum(precisions) / len(precisions) if precisions else 0

		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query
		"""
		recall = -1

		#Fill in code here
		top_k = query_doc_IDs_ordered[:k]
		true_set = set(true_doc_IDs)
		relevant_count = sum(1 for doc_id in top_k if doc_id in true_set)
		recall = relevant_count / len(true_set) if len(true_set) > 0 else 0

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""
		meanRecall = -1

		#Fill in code here
		relevance = {}
		for qrel in qrels:
			query_num = int(qrel["query_num"])
			doc_id = int(qrel["id"])
			if query_num not in relevance:
				relevance[query_num] = []
			relevance[query_num].append(doc_id)

		recalls = []
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = relevance.get(query_id, [])
			recall = self.queryRecall(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			recalls.append(recall)

		meanRecall = sum(recalls) / len(recalls) if recalls else 0

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query
		"""
		fscore = -1

		#Fill in code here
		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		if precision + recall > 0:
			# F0.5 score: beta = 0.5. Formula: (1 + beta^2) * P * R / (beta^2 * P + R)
			# beta^2 = 0.25
			fscore = (1.25 * precision * recall) / (0.25 * precision + recall)
		else:
			fscore = 0

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""
		meanFscore = -1

		#Fill in code here
		relevance = {}
		for qrel in qrels:
			query_num = int(qrel["query_num"])
			doc_id = int(qrel["id"])
			if query_num not in relevance:
				relevance[query_num] = []
			relevance[query_num].append(doc_id)

		fscores = []
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = relevance.get(query_id, [])
			fscore = self.queryFscore(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			fscores.append(fscore)

		meanFscore = sum(fscores) / len(fscores) if fscores else 0

		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query
		"""
		nDCG = -1

		#Fill in code here
		# true_doc_IDs here is expected to be a dict: {doc_id: relevance_score}
		# Compute DCG for the ranked list
		top_k = query_doc_IDs_ordered[:k]
		dcg = 0
		for i, doc_id in enumerate(top_k):
			rel = true_doc_IDs.get(doc_id, 0)
			if i == 0:
				dcg += rel
			else:
				dcg += rel / math.log2(i + 1)

		# Compute ideal DCG (IDCG): sort all relevance scores descending
		ideal_rels = sorted(true_doc_IDs.values(), reverse=True)[:k]
		idcg = 0
		for i, rel in enumerate(ideal_rels):
			if i == 0:
				idcg += rel
			else:
				idcg += rel / math.log2(i + 1)

		nDCG = dcg / idcg if idcg > 0 else 0

		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""
		meanNDCG = -1

		#Fill in code here
		# Build relevance mapping: query_num -> {doc_id: relevance_score}
		# Relevance = 5 - position (so position 1 -> 4, position 4 -> 1)
		relevance = {}
		for qrel in qrels:
			query_num = int(qrel["query_num"])
			doc_id = int(qrel["id"])
			position = int(qrel["position"])
			rel_score = 5 - position
			if query_num not in relevance:
				relevance[query_num] = {}
			relevance[query_num][doc_id] = rel_score

		ndcgs = []
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = relevance.get(query_id, {})
			ndcg = self.queryNDCG(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			ndcgs.append(ndcg)

		meanNDCG = sum(ndcgs) / len(ndcgs) if ndcgs else 0

		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)
		"""
		avgPrecision = -1

		#Fill in code here
		top_k = query_doc_IDs_ordered[:k]
		true_set = set(true_doc_IDs)
		relevant_count = 0
		precision_sum = 0

		for i, doc_id in enumerate(top_k):
			if doc_id in true_set:
				relevant_count += 1
				precision_at_i = relevant_count / (i + 1)
				precision_sum += precision_at_i

		avgPrecision = precision_sum / len(true_set) if len(true_set) > 0 else 0

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries
		"""
		meanAveragePrecision = -1

		#Fill in code here
		relevance = {}
		for qrel in q_rels:
			query_num = int(qrel["query_num"])
			doc_id = int(qrel["id"])
			if query_num not in relevance:
				relevance[query_num] = []
			relevance[query_num].append(doc_id)

		avg_precisions = []
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = relevance.get(query_id, [])
			ap = self.queryAveragePrecision(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			avg_precisions.append(ap)

		meanAveragePrecision = sum(avg_precisions) / len(avg_precisions) if avg_precisions else 0

		return meanAveragePrecision



	def queryReciprocalRank(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of reciprocal rank for a single query

		Parameters
		----------
		arg1 : list
			Ranked list of document IDs
		arg2 : int
			Query ID
		arg3 : list
			List of relevant document IDs
		arg4 : int
			The k value

		Returns
		-------
		float
			Reciprocal rank value
		"""

		reciprocalRank = -1

		#Fill in code here
		top_k = query_doc_IDs_ordered[:k]
		true_set = set(true_doc_IDs)
		reciprocalRank = 0
		for i, doc_id in enumerate(top_k):
			if doc_id in true_set:
				reciprocalRank = 1.0 / (i + 1)
				break

		return reciprocalRank


	def meanReciprocalRank(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of Mean Reciprocal Rank (MRR)
		averaged over all queries

		Parameters
		----------
		arg1 : list
			List of ranked document lists
		arg2 : list
			Query IDs
		arg3 : list
			Relevance judgments
		arg4 : int
			The k value

		Returns
		-------
		float
			MRR value
		"""

		meanReciprocalRank = -1

		#Fill in code here
		relevance = {}
		for qrel in qrels:
			query_num = int(qrel["query_num"])
			doc_id = int(qrel["id"])
			if query_num not in relevance:
				relevance[query_num] = []
			relevance[query_num].append(doc_id)

		reciprocal_ranks = []
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = relevance.get(query_id, [])
			rr = self.queryReciprocalRank(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			reciprocal_ranks.append(rr)

		meanReciprocalRank = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0

		return meanReciprocalRank
