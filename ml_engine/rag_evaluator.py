# rag_evaluator.py — FINAL ELITE v3 (ROBUST + CALIBRATED + AUDITABLE)

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

EPS = 1e-9

STOPWORDS = {
    "the","is","and","of","to","a","in","that","it","on","for"
}

MAX_TOKENS = 500


# ─────────────────────────────────────────────
# TOKENIZATION (SAFE + BOUNDED)
# ─────────────────────────────────────────────
def _tokenize(text: str) -> List[str]:
    if not isinstance(text, str) or not text:
        return []

    text = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = text.split()

    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]

    return tokens[:MAX_TOKENS]


def _safe_div(a, b):
    return a / b if b else 0.0


# ─────────────────────────────────────────────
# TF COSINE (IMPROVED SEMANTICS)
# ─────────────────────────────────────────────
def _tf_vector(tokens: List[str]) -> Dict[str, float]:
    if not tokens:
        return {}

    counts = Counter(tokens)
    total = sum(counts.values()) + EPS

    return {k: v / total for k, v in counts.items()}


def _cosine_tf(a_tokens, b_tokens):
    if not a_tokens or not b_tokens:
        return 0.0

    vec_a = _tf_vector(a_tokens)
    vec_b = _tf_vector(b_tokens)

    common = set(vec_a) & set(vec_b)

    num = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = np.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = np.sqrt(sum(v * v for v in vec_b.values()))

    denom = norm_a * norm_b + EPS

    return float(num / denom)


# ─────────────────────────────────────────────
# GROUNDING
# ─────────────────────────────────────────────
def compute_groundedness(answer, context_docs):
    ctx = " ".join(context_docs or [])
    return _cosine_tf(_tokenize(answer), _tokenize(ctx))


# ─────────────────────────────────────────────
# HALLUCINATION (SOFT + LENGTH NORMALIZED)
# ─────────────────────────────────────────────
def hallucination_score(answer, context_docs):

    answer_tokens = _tokenize(answer)
    context_tokens = set(_tokenize(" ".join(context_docs or [])))

    if not answer_tokens:
        return 1.0

    unsupported = sum(1 for t in answer_tokens if t not in context_tokens)

    raw = unsupported / (len(answer_tokens) + EPS)

    # soften penalty (avoid punishing paraphrases too harshly)
    return float(np.clip(raw ** 0.7, 0.0, 1.0))


# ─────────────────────────────────────────────
# RELEVANCE
# ─────────────────────────────────────────────
def answer_relevance(query, answer):
    return _cosine_tf(_tokenize(query), _tokenize(answer))


# ─────────────────────────────────────────────
# COVERAGE
# ─────────────────────────────────────────────
def context_coverage(answer, context_docs):

    if not context_docs:
        return 0.0

    scores = [
        _cosine_tf(_tokenize(answer), _tokenize(doc))
        for doc in context_docs if doc
    ]

    return float(np.mean(scores)) if scores else 0.0


# ─────────────────────────────────────────────
# CONSISTENCY (REPETITION CONTROL)
# ─────────────────────────────────────────────
def consistency_score(answer):

    tokens = _tokenize(answer)

    if not tokens:
        return 0.0

    counts = Counter(tokens)
    repetition = sum(v - 1 for v in counts.values() if v > 1)

    penalty = repetition / (len(tokens) + EPS)

    return float(np.clip(1 - penalty, 0.0, 1.0))


# ─────────────────────────────────────────────
# DECISION
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
# CONFIDENCE CALIBRATION (STABLE)
# ─────────────────────────────────────────────
def _calibrate_confidence(base_conf, metrics):

    base_conf = float(np.clip(base_conf, 0.0, 1.0))

    penalty = (
        (1 - metrics["groundedness"]) * 0.35 +
        metrics["hallucination"] * 0.35 +
        (1 - metrics["relevance"]) * 0.2 +
        (1 - metrics["coverage"]) * 0.1
    )

    adjusted = base_conf * (1 - penalty)

    return float(np.clip(adjusted, 0.0, 1.0))


# ─────────────────────────────────────────────
# MAIN
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
            "groundedness": float(np.clip(grounded, 0.0, 1.0)),
            "hallucination": float(np.clip(halluc, 0.0, 1.0)),
            "relevance": float(np.clip(relevance, 0.0, 1.0)),
            "coverage": float(np.clip(coverage, 0.0, 1.0)),
            "consistency": float(np.clip(consistency, 0.0, 1.0)),
        }

        decision = _safety_decision(metrics)

        calibrated_conf = _calibrate_confidence(base_confidence, metrics)

        composite = (
            0.3 * metrics["groundedness"] +
            0.2 * (1 - metrics["hallucination"]) +
            0.2 * metrics["relevance"] +
            0.15 * metrics["coverage"] +
            0.15 * metrics["consistency"]
        )

        result = {
            **{k: round(v, 4) for k, v in metrics.items()},
            "composite_score": round(float(composite), 4),
            "decision": decision,
            "confidence_adjusted": round(calibrated_conf, 3),
        }

        # 🔍 audit trace (critical for debugging)
        result["_audit"] = {
            "answer_len": len(answer or ""),
            "num_docs": len(context_docs or []),
            "token_count": len(_tokenize(answer))
        }

        return result

    except Exception as e:
        logger.error(f"[RAG Eval][FAIL] {e}")

        return {
            "groundedness": 0.0,
            "hallucination": 1.0,
            "relevance": 0.0,
            "coverage": 0.0,
            "consistency": 0.0,
            "composite_score": 0.0,
            "decision": "unsafe",
            "confidence_adjusted": 0.0,
            "_audit": {"error": True}
        }