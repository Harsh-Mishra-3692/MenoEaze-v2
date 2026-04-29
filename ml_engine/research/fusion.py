# ml_engine/research/fusion.py — ELITE v3 (ROBUST + RESEARCH-GRADE)

import logging
from typing import List, Dict, Any

logger = logging.getLogger("menoeaze.fusion")

EPS = 1e-8


# ─────────────────────────────────────────────
# SAFE DOC ID
# ─────────────────────────────────────────────
def _get_doc_id(doc: Dict[str, Any]) -> str:
    """
    Robust ID extraction with fallback to content hash
    """
    if "id" in doc and doc["id"]:
        return str(doc["id"])

    content = doc.get("content", "")
    return str(hash(content[:200]))


# ─────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────
def _normalize(scores: List[float]) -> List[float]:
    if not scores:
        return scores

    min_s, max_s = min(scores), max(scores)

    if max_s == min_s:
        return [0.5] * len(scores)

    return [(s - min_s) / (max_s - min_s + EPS) for s in scores]


# ─────────────────────────────────────────────
# RECIPROCAL RANK FUSION (FIXED)
# ─────────────────────────────────────────────
def reciprocal_rank_fusion(
    results_list: List[List[Dict[str, Any]]],
    k: int = 60,
    top_k: int = 8
) -> List[Dict[str, Any]]:

    if not results_list:
        return []

    scores = {}
    doc_store = {}

    for results in results_list:
        for rank, doc in enumerate(results):
            doc_id = _get_doc_id(doc)

            doc_store[doc_id] = doc

            score = 1.0 / (k + rank)
            scores[doc_id] = scores.get(doc_id, 0.0) + score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    final_docs = []
    for doc_id, score in ranked[:top_k]:
        d = doc_store[doc_id]
        d["fusion_score"] = float(score)
        final_docs.append(d)

    logger.info(f"[Fusion:RRF] docs={len(final_docs)}")

    return final_docs


# ─────────────────────────────────────────────
# WEIGHTED SCORE FUSION (UPGRADED)
# ─────────────────────────────────────────────
def weighted_score_fusion(
    vector_docs: List[Dict[str, Any]],
    keyword_docs: List[Dict[str, Any]],
    alpha: float = 0.7,
    beta: float = 0.3,
    top_k: int = 8
) -> List[Dict[str, Any]]:

    if not vector_docs and not keyword_docs:
        return []

    merged = {}
    doc_store = {}

    # collect vector scores
    vec_scores = [d.get("similarity", 0.0) for d in vector_docs]
    vec_norm = _normalize(vec_scores)

    for d, score in zip(vector_docs, vec_norm):
        doc_id = _get_doc_id(d)

        doc_store[doc_id] = d

        merged[doc_id] = merged.get(doc_id, 0.0)
        merged[doc_id] += alpha * score

    # collect keyword scores
    kw_scores = [d.get("keyword_score", 0.0) for d in keyword_docs]
    kw_norm = _normalize(kw_scores)

    for d, score in zip(keyword_docs, kw_norm):
        doc_id = _get_doc_id(d)

        doc_store[doc_id] = d

        merged[doc_id] = merged.get(doc_id, 0.0)
        merged[doc_id] += beta * score

    # sort
    ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)

    final_docs = []
    for doc_id, score in ranked[:top_k]:
        d = doc_store[doc_id]
        d["fusion_score"] = float(score)
        final_docs.append(d)

    logger.info(f"[Fusion:Weighted] docs={len(final_docs)}")

    return final_docs


# ─────────────────────────────────────────────
# HYBRID FUSION (NEW - RESEARCH USE)
# ─────────────────────────────────────────────
def hybrid_fusion(
    vector_docs: List[Dict[str, Any]],
    keyword_docs: List[Dict[str, Any]],
    method: str = "weighted",
    top_k: int = 8
) -> List[Dict[str, Any]]:
    """
    Unified interface for experiments
    """

    if method == "rrf":
        return reciprocal_rank_fusion(
            [vector_docs, keyword_docs],
            top_k=top_k
        )

    return weighted_score_fusion(
        vector_docs,
        keyword_docs,
        top_k=top_k
    )