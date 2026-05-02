# reranker.py — FINAL ELITE v4 (CLINICAL + SAFE + SYSTEM-AWARE)

import logging
import time
import re
from typing import List, Dict, Any

import torch
import numpy as np
from sentence_transformers import CrossEncoder

logger = logging.getLogger("menoeaze.reranker")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

MAX_DOCS = 10
TOP_K = 3
MAX_TEXT_LENGTH = 512
MAX_QUERY_LENGTH = 256

BATCH_SIZE = 8
TIMEOUT_SEC = 5.0

# weights (system-aligned)
ALPHA = 0.55   # reranker
BETA = 0.20    # vector similarity
GAMMA = 0.10   # hybrid score
DELTA = 0.15   # priority boost

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None


# ─────────────────────────────────────────────
# MODEL LOAD
# ─────────────────────────────────────────────
def _get_model():
    global _model

    if _model is not None:
        return _model

    try:
        logger.info("[RERANKER] loading model...")
        _model = CrossEncoder(MODEL_NAME, device=DEVICE)
        logger.info(f"[RERANKER] loaded on {DEVICE}")
    except Exception as e:
        logger.error(f"[RERANKER] load failed: {e}")
        _model = None

    return _model


# ─────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────
def _clean(text: str, max_len: int) -> str:
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text.strip())
    return text[:max_len]


# ─────────────────────────────────────────────
# QUALITY FILTER
# ─────────────────────────────────────────────
def _is_valid_doc(text: str) -> bool:
    if not text or len(text) < 30:
        return False

    alpha_ratio = sum(c.isalpha() for c in text) / len(text)
    return alpha_ratio > 0.6


# ─────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────
def _normalize(scores: List[float]) -> List[float]:
    if not scores:
        return scores

    mn, mx = min(scores), max(scores)

    if mx - mn < 1e-6:
        return [0.5] * len(scores)

    return [(s - mn) / (mx - mn) for s in scores]


# ─────────────────────────────────────────────
# DIVERSITY FILTER
# ─────────────────────────────────────────────
def _diversify(docs: List[Dict[str, Any]], top_k: int):
    seen = set()
    result = []

    for d in docs:
        src = d.get("document_name")

        if src in seen:
            continue

        seen.add(src)
        result.append(d)

        if len(result) >= top_k:
            break

    return result


# ─────────────────────────────────────────────
# FALLBACK
# ─────────────────────────────────────────────
def _fallback_rank(docs, top_k):
    logger.warning("[RERANKER] fallback")

    for d in docs:
        sim = d.get("similarity", 0.5)
        d["final_score"] = sim

    docs.sort(key=lambda x: x["final_score"], reverse=True)
    return docs[:top_k]


# ─────────────────────────────────────────────
# MAIN RERANK
# ─────────────────────────────────────────────
def rerank(
    query: str,
    docs: List[Dict[str, Any]],
    top_k: int = TOP_K,
    return_debug: bool = False
):

    if not query or not docs:
        return []

    model = _get_model()

    # ── sanitize
    query = _clean(query, MAX_QUERY_LENGTH)

    # ── filter docs
    filtered = []
    for d in docs[:MAX_DOCS]:
        text = _clean(d.get("content", ""), MAX_TEXT_LENGTH)

        if not _is_valid_doc(text):
            continue

        d["content"] = text
        filtered.append(d)

    if not filtered:
        return []

    if model is None:
        return _fallback_rank(filtered, top_k)

    start = time.time()

    try:
        pairs = [(query, d["content"]) for d in filtered]

        raw_scores = model.predict(
            pairs,
            batch_size=BATCH_SIZE,
            show_progress_bar=False
        )

        raw_scores = list(raw_scores)
        norm_scores = _normalize(raw_scores)

        for d, r_score, raw in zip(filtered, norm_scores, raw_scores):

            sim = d.get("similarity", 0.0)
            hybrid = d.get("hybrid_score", sim)
            priority = d.get("priority", 0.0)

            final = (
                ALPHA * r_score +
                BETA * sim +
                GAMMA * hybrid +
                DELTA * priority
            )

            d.update({
                "rerank_score": float(r_score),
                "final_score": float(final),
                "cross_raw": float(raw)
            })

        # ── sort
        filtered.sort(key=lambda x: x["final_score"], reverse=True)

        # ── diversify
        results = _diversify(filtered, top_k)

        latency = (time.time() - start) * 1000

        logger.info(f"[RERANKER] {len(results)} docs | {latency:.1f}ms")

        if return_debug:
            return {
                "results": results,
                "latency": latency,
                "raw_scores": raw_scores
            }

        return results

    except Exception as e:
        logger.error(f"[RERANKER] failed: {e}")
        return _fallback_rank(filtered, top_k)