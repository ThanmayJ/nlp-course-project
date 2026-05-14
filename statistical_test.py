import json
import os
import numpy as np
from scipy import stats
from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval
from improvedInformationRetrieval import ImprovedInformationRetrieval
from evaluation import Evaluation

def load_data(dataset_path):
    queries_json = json.load(open(os.path.join(dataset_path, "cran_queries.json"), 'r'))
    query_ids = [item["query number"] for item in queries_json]
    queries = [item["query"] for item in queries_json]

    docs_json = json.load(open(os.path.join(dataset_path, "cran_docs.json"), 'r'))
    doc_ids = [item["id"] for item in docs_json]
    docs = [item["body"] for item in docs_json]
    
    qrels = json.load(open(os.path.join(dataset_path, "cran_qrels.json"), 'r'))
    return queries, query_ids, docs, doc_ids, qrels

def preprocess(texts):
    # Minimal pipeline
    seg = SentenceSegmentation()
    tok = Tokenization()
    inf = InflectionReduction()
    stop = StopwordRemoval()
    
    processed = []
    for text in texts:
        s = seg.punkt(text)
        t = tok.pennTreeBank(s)
        i = inf.reduce(t)
        st = stop.fromList(i)
        processed.append(st)
    return processed

def get_ap_scores(ir_system, processed_docs, doc_ids, processed_queries, query_ids, qrels, eval_engine, k=10):
    ir_system.buildIndex(processed_docs, doc_ids)
    ordered = ir_system.rank(processed_queries)
    
    relevance = {}
    for qrel in qrels:
        qn = int(qrel["query_num"])
        did = int(qrel["id"])
        if qn not in relevance:
            relevance[qn] = []
        relevance[qn].append(did)
        
    ap_scores = []
    for i, q_id in enumerate(query_ids):
        true_ids = relevance.get(q_id, [])
        ap = eval_engine.queryAveragePrecision(ordered[i], q_id, true_ids, k)
        ap_scores.append(ap)
    return np.array(ap_scores), ordered

def main():
    print("Running Statistical Hypothesis Test comparing VSM vs LSA...")
    queries, query_ids, docs, doc_ids, qrels = load_data("cranfield")
    p_queries = preprocess(queries)
    p_docs = preprocess(docs)
    
    eval_engine = Evaluation()
    
    # Base VSM
    vsm = InformationRetrieval()
    print("\nProcessing Base VSM...")
    ap_vsm, ordered_vsm = get_ap_scores(vsm, p_docs, doc_ids, p_queries, query_ids, qrels, eval_engine)
    
    # Improved LSA
    lsa = ImprovedInformationRetrieval(n_components=100)
    print("\nProcessing Improved LSA...")
    ap_lsa, ordered_lsa = get_ap_scores(lsa, p_docs, doc_ids, p_queries, query_ids, qrels, eval_engine)
    
    mean_vsm = np.mean(ap_vsm)
    mean_lsa = np.mean(ap_lsa)
    
    print(f"\n--- Summary Statistics (MAP@10) ---")
    print(f"Baseline VSM MAP: {mean_vsm:.4f}")
    print(f"Improved LSA MAP: {mean_lsa:.4f}")
    
    # Perform Paired t-test (Since it is the same queries)
    t_stat, p_val = stats.ttest_rel(ap_lsa, ap_vsm)
    
    print(f"\n--- Statistical Test ---")
    print(f"H0: No difference between LSA and VSM")
    print(f"H1: LSA outperforms or differs significantly from VSM")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_val:.4e}")
    
    alpha = 0.05
    if p_val < alpha:
        print(f"Result: Significant at alpha={alpha}. Reject H0.")
    else:
        print(f"Result: Not statistically significant at alpha={alpha}. Fail to reject H0.")

    # Look for concrete examples of VSM failure fixed by LSA for Section 4 & 5 report
    # Find a query where VSM performed very poorly (AP=0) and LSA did better
    differences = ap_lsa - ap_vsm
    best_imp_idx = np.argmax(differences)
    worst_imp_idx = np.argmin(differences)
    
    print("\n--- Example for Qualitative Analysis ---")
    print(f"Query ID: {query_ids[best_imp_idx]}")
    print(f"Query Text: '{queries[best_imp_idx]}'")
    print(f"VSM AP: {ap_vsm[best_imp_idx]:.4f} | LSA AP: {ap_lsa[best_imp_idx]:.4f}")

if __name__ == "__main__":
    main()
