import numpy as np
from typing import List, Dict, Any


def compute_cosine_similarity(query_vec: np.ndarray, chunk_vecs: np.ndarray) -> np.ndarray:
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)

    dot_product = np.dot(chunk_vecs, query_vec.T).squeeze(axis=1)

    query_norm = np.linalg.norm(query_vec)
    chunk_norms = np.linalg.norm(chunk_vecs, axis=1)

    denom = chunk_norms * query_norm
    denom[denom == 0] = 1.0

    scores = dot_product / denom
    return np.clip(scores, -1.0, 1.0)


def retrieve_top_k(
    query_vec: np.ndarray,
    chunk_vecs: np.ndarray,
    chunks: List[Dict[str, Any]],
    top_k: int = 3,
    threshold: float = 0.25
) -> Dict[str, Any]:
    """
    Retrieves the top-k most similar chunks for a query using cosine similarity.
    """
    if len(chunks) == 0 or chunk_vecs.size == 0:
        return {
            "results": [],
            "is_relevant": False,
            "max_score": 0.0
        }

    scores = compute_cosine_similarity(query_vec, chunk_vecs)

    ranked_indices = np.argsort(scores)[::-1]

    top_indices = ranked_indices[:min(top_k, len(chunks))]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        results.append({
            "chunk": chunks[idx],
            "score": score
        })

    max_score = float(scores[ranked_indices[0]]) if len(scores) > 0 else 0.0
    is_relevant = max_score >= threshold

    return {
        "results": results,
        "is_relevant": is_relevant,
        "max_score": max_score
    }
