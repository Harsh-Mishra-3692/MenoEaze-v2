# rag_evaluator.py — ELITE (SEMANTIC + ROBUST RAG EVALUATION)

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter

logger = logging.getLogger("menoeaze.rag_eval")

# ─────────────────────────────────────────────
# TEXT UTILITIES
# ─────────────────────────────────────────────
def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return text.lower().split()


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


# ─────────────────────────────────────────────
# LIGHTWEIGHT SEMANTIC SIMILARITY
# (fallback without external models)
# ─────────────────────────────────────────────
def _cosine_similarity(a_tokens: List[str], b_tokens: List[str]) -> float:
    """
    Bag-of-words cosine similarity (lightweight fallback).
    """
    if not a_tokens or not b_tokens:
        return 0.0

    vocab = list(set(a_tokens + b_tokens))
    vec_a = np.array([a_tokens.count(t) for t in vocab])
    vec_b = np.array([b_tokens.count(t) for t in vocab])

    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / denom)


# ─────────────────────────────────────────────
# RETRIEVAL METRICS
# ─────────────────────────────────────────────
def ndcg_at_k(relevant: List[str], retrieved: List[str], k: int = 5) -> float:
    def dcg(items):
        return sum(
            (1 if item in relevant else 0) / np.log2(i + 2)
            for i, item in enumerate(items[:k])
        )

    ideal = dcg(relevant[:k])
    actual = dcg(retrieved[:k])

    return _safe_div(actual, ideal)


def mean_reciprocal_rank(relevant: List[str], retrieved: List[str]) -> float:
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1 / (i + 1)
    return 0.0


# ─────────────────────────────────────────────
# SEMANTIC GROUNDING
# ─────────────────────────────────────────────
def compute_groundedness(answer: str, context_docs: List[str]) -> float:
    if not answer or not context_docs:
        return 0.0

    answer_tokens = _tokenize(answer)
    context_tokens = _tokenize(" ".join(context_docs))

    return _cosine_similarity(answer_tokens, context_tokens)


# ─────────────────────────────────────────────
# HALLUCINATION SCORE
# ─────────────────────────────────────────────
def hallucination_score(answer: str, context_docs: List[str]) -> float:
    """
    Higher → more hallucination
    """

    if not answer:
        return 1.0

    answer_tokens = _tokenize(answer)
    context_tokens = set(_tokenize(" ".join(context_docs)))

    unsupported = [t for t in answer_tokens if t not in context_tokens]

    return _safe_div(len(unsupported), len(answer_tokens))


# ─────────────────────────────────────────────
# QUERY RELEVANCE (SEMANTIC)
# ─────────────────────────────────────────────
def answer_relevance(query: str, answer: str) -> float:
    if not query or not answer:
        return 0.0

    return _cosine_similarity(_tokenize(query), _tokenize(answer))


# ─────────────────────────────────────────────
# CONTEXT COVERAGE
# ─────────────────────────────────────────────
def context_coverage(answer: str, context_docs: List[str]) -> float:
    if not answer or not context_docs:
        return 0.0

    answer_tokens = _tokenize(answer)

    scores = [
        _cosine_similarity(answer_tokens, _tokenize(doc))
        for doc in context_docs
    ]

    return float(np.mean(scores)) if scores else 0.0


# ─────────────────────────────────────────────
# CONSISTENCY (REDUNDANCY PENALTY)
# ─────────────────────────────────────────────
def consistency_score(answer: str) -> float:
    tokens = _tokenize(answer)

    if not tokens:
        return 0.0

    counts = Counter(tokens)
    repeated = sum(c for c in counts.values() if c > 1)

    repetition_ratio = repeated / len(tokens)

    return float(np.clip(1 - repetition_ratio, 0.0, 1.0))


# ─────────────────────────────────────────────
# FINAL RAG EVALUATION
# ─────────────────────────────────────────────
def evaluate_rag(
    query: str,
    answer: str,
    retrieved_docs: List[str],
    context_docs: List[str],
    relevant_docs: Optional[List[str]] = None
) -> Dict[str, Any]:

    try:
        results = {}

        # ── Retrieval metrics ─────────────────
        if relevant_docs:
            results["ndcg@5"] = round(ndcg_at_k(relevant_docs, retrieved_docs, 5), 4)
            results["mrr"] = round(mean_reciprocal_rank(relevant_docs, retrieved_docs), 4)

        # ── Generation metrics ────────────────
        grounded = compute_groundedness(answer, context_docs)
        halluc = hallucination_score(answer, context_docs)
        relevance = answer_relevance(query, answer)
        coverage = context_coverage(answer, context_docs)
        consistency = consistency_score(answer)

        results.update({
            "groundedness": round(grounded, 4),
            "hallucination": round(halluc, 4),
            "relevance": round(relevance, 4),
            "coverage": round(coverage, 4),
            "consistency": round(consistency, 4),
        })

        # ── Composite score (balanced)
        composite = (
            0.30 * grounded +
            0.20 * (1 - halluc) +
            0.20 * relevance +
            0.15 * coverage +
            0.15 * consistency
        )

        results["composite_score"] = round(float(composite), 4)

        return results

    except Exception as e:
        logger.error(f"[RAG Eval] Failed: {e}")
        return {
            "groundedness": 0.0,
            "hallucination": 1.0,
            "relevance": 0.0,
            "coverage": 0.0,
            "consistency": 0.0,
            "composite_score": 0.0,
        }
