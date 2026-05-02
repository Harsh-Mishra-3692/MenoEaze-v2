# rag_evaluator.py — FINAL ELITE (CLINICAL + RUNTIME SAFE)

import logging
import numpy as np
import re
from typing import List, Dict, Any, Optional
from collections import Counter

logger = logging.getLogger("menoeaze.rag_eval")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MIN_GROUNDEDNESS = 0.2
MAX_HALLUCINATION = 0.6
MIN_RELEVANCE = 0.15

STOPWORDS = {
    "the","is","and","of","to","a","in","that","it","on","for"
}


# ─────────────────────────────────────────────
# TEXT UTILITIES (ROBUST)
# ─────────────────────────────────────────────
def _tokenize(text: str) -> List[str]:
    if not text:
        return []

    text = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = text.split()

    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def _safe_div(a, b):
    return a / b if b else 0.0


# ─────────────────────────────────────────────
# SEMANTIC SIMILARITY (IMPROVED)
# ─────────────────────────────────────────────
def _cosine(a, b):
    if not a or not b:
        return 0.0

    vocab = list(set(a + b))
    vec_a = np.array([a.count(t) for t in vocab])
    vec_b = np.array([b.count(t) for t in vocab])

    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return float(np.dot(vec_a, vec_b) / denom) if denom else 0.0


# ─────────────────────────────────────────────
# GROUNDING
# ─────────────────────────────────────────────
def compute_groundedness(answer, context_docs):
    return _cosine(
        _tokenize(answer),
        _tokenize(" ".join(context_docs))
    )


# ─────────────────────────────────────────────
# HALLUCINATION (IMPROVED)
# ─────────────────────────────────────────────
def hallucination_score(answer, context_docs):

    answer_tokens = _tokenize(answer)
    context_tokens = set(_tokenize(" ".join(context_docs)))

    if not answer_tokens:
        return 1.0

    unsupported = sum(1 for t in answer_tokens if t not in context_tokens)

    return unsupported / len(answer_tokens)


# ─────────────────────────────────────────────
# RELEVANCE
# ─────────────────────────────────────────────
def answer_relevance(query, answer):
    return _cosine(_tokenize(query), _tokenize(answer))


# ─────────────────────────────────────────────
# COVERAGE
# ─────────────────────────────────────────────
def context_coverage(answer, context_docs):

    scores = [
        _cosine(_tokenize(answer), _tokenize(doc))
        for doc in context_docs
    ]

    return float(np.mean(scores)) if scores else 0.0


# ─────────────────────────────────────────────
# CONSISTENCY
# ─────────────────────────────────────────────
def consistency_score(answer):

    tokens = _tokenize(answer)
    if not tokens:
        return 0.0

    counts = Counter(tokens)
    repetition = sum(v for v in counts.values() if v > 1)

    return float(np.clip(1 - repetition / len(tokens), 0.0, 1.0))


# ─────────────────────────────────────────────
# DECISION LAYER (NEW)
# ─────────────────────────────────────────────
def _safety_decision(metrics):

    grounded = metrics["groundedness"]
    halluc = metrics["hallucination"]
    relevance = metrics["relevance"]

    if grounded < MIN_GROUNDEDNESS:
        return "unsafe"

    if halluc > MAX_HALLUCINATION:
        return "unsafe"

    if relevance < MIN_RELEVANCE:
        return "weak"

    return "safe"


# ─────────────────────────────────────────────
# CONFIDENCE CALIBRATION (NEW)
# ─────────────────────────────────────────────
def _calibrate_confidence(base_conf, metrics):

    penalty = (
        (1 - metrics["groundedness"]) * 0.4 +
        metrics["hallucination"] * 0.4 +
        (1 - metrics["relevance"]) * 0.2
    )

    return float(np.clip(base_conf * (1 - penalty), 0.0, 1.0))


# ─────────────────────────────────────────────
# MAIN EVALUATOR
# ─────────────────────────────────────────────
def evaluate_rag(
    query: str,
    answer: str,
    retrieved_docs: List[str],
    context_docs: List[str],
    relevant_docs: Optional[List[str]] = None,
    base_confidence: float = 0.5
) -> Dict[str, Any]:

    try:
        grounded = compute_groundedness(answer, context_docs)
        halluc = hallucination_score(answer, context_docs)
        relevance = answer_relevance(query, answer)
        coverage = context_coverage(answer, context_docs)
        consistency = consistency_score(answer)

        metrics = {
            "groundedness": grounded,
            "hallucination": halluc,
            "relevance": relevance,
            "coverage": coverage,
            "consistency": consistency,
        }

        # ── decision layer
        decision = _safety_decision(metrics)

        # ── confidence recalibration
        calibrated_conf = _calibrate_confidence(base_confidence, metrics)

        # ── composite score
        composite = (
            0.3 * grounded +
            0.2 * (1 - halluc) +
            0.2 * relevance +
            0.15 * coverage +
            0.15 * consistency
        )

        return {
            **{k: round(v, 4) for k, v in metrics.items()},
            "composite_score": round(composite, 4),
            "decision": decision,
            "confidence_adjusted": round(calibrated_conf, 3)
        }

    except Exception as e:
        logger.error(f"[RAG Eval] fatal: {e}")

        return {
            "groundedness": 0.0,
            "hallucination": 1.0,
            "relevance": 0.0,
            "coverage": 0.0,
            "consistency": 0.0,
            "composite_score": 0.0,
            "decision": "unsafe",
            "confidence_adjusted": 0.0
        }