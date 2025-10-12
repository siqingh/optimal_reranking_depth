import pyterrier as pt
import numpy as np
import pandas as pd

def make_equal_content_bins(data, n_bins):
    """
    Create equal-content bins (quantile-based).
    
    Args:
        data (array-like): Input numeric data.
        n_bins (int): Number of bins to split into.
    
    Returns:
        bin_edges (np.ndarray): The bin boundaries.
    """
    data = np.asarray(data)
    # Get quantile edges
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(data, quantiles)
    # Ensure unique edges (drop duplicates if data has ties)
    bin_edges = np.unique(bin_edges)
    return bin_edges

def assign_to_bin(val, bin_edges):
    """
    Assign a single value to a bin given bin edges.
    
    Args:
        val (float): Value to assign.
        bin_edges (array): Bin boundaries.
    
    Returns:
        str: Label like "low-high".
    """
    idx = np.searchsorted(bin_edges, val, side="right") - 1
    # Clamp idx to valid range
    idx = max(0, min(idx, len(bin_edges) - 2))
    return f"{bin_edges[idx]:.2f}~{bin_edges[idx+1]:.2f}"

def load_umbrela_scores():
    umbrela_scores = pd.read_csv('data/all_umbrela_judgements.tsv', sep='\t')
    umbrela_scores = umbrela_scores.drop(['query_text', 'doc_text'], axis=1)
    umbrela_scores['query_id'] = umbrela_scores['query_id'].astype(str)
    umbrela_scores['doc_id'] = umbrela_scores['doc_id'].astype(str)
    return umbrela_scores

def convert_run(path):
    run = pt.io.read_results(path)
    run['score'] = run['rank'].apply(lambda x: 1. / (1. + x))
    run = run.drop(['rank', 'name'], axis=1)
    run = run.set_axis(['query_id', 'doc_id', 'score'], axis='columns')
    run['query_id'] = run['query_id'].astype(str)
    run['doc_id'] = run['doc_id'].astype(str)
    return run

def dcg(relevances, k=10):
    relevances = np.asarray(relevances, dtype=np.float64)[:k]
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return np.sum(relevances / discounts)

def dcg_from_run(run_df, qrel_df, k=10):
    # Create a lookup for (query_id, doc_id) → relevance
    relevance_lookup = {
        (qid, did): rel
        for qid, did, rel in qrel_df[['query_id', 'doc_id', 'relevance']].itertuples(index=False)
    }

    dcg_scores = {}

    # Process each query
    for qid, group in run_df.groupby('query_id'):
        # Sort docs by score descending
        top_docs = group.sort_values('score', ascending=False).head(k)

        # Get list of relevance scores in run order
        relevances = [
            relevance_lookup.get((qid, doc_id), 0)
            for doc_id in top_docs['doc_id']
        ]

        dcg_scores[qid] = float(dcg(relevances, k))

    return dcg_scores

def dcg_for_query(run_df, qrel_df, query_id, k=10):
    qrel_lookup = {
        did: rel
        for did, rel in qrel_df.loc[qrel_df['query_id'] == str(query_id), ['doc_id', 'relevance']].itertuples(index=False)
    }
    group = run_df[run_df['query_id'] == str(query_id)]

    # Sort docs by score descending, take top-k
    top_docs = group.sort_values('score', ascending=False).head(k)

    missing_docs = []
    relevances = []

    # Get list of relevance scores in run order
    for doc_id in top_docs['doc_id']:
        if doc_id not in qrel_lookup:
            missing_docs.append(doc_id)
        else:
            relevances.append(qrel_lookup[doc_id])

    return float(dcg(relevances, k)), missing_docs

def prettify_label(text: str) -> str:
    """Split by underscore and capitalize each word."""
    if '_' in text:
        return " ".join(word.capitalize() for word in text.split("_"))
    return text