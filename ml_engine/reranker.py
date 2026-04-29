# reranker.py — ELITE v3 (PRODUCTION + RESEARCH + SAFE)

import logging
import time
from typing import List, Dict, Any

import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger("menoeaze.reranker")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

MAX_DOCS = 8
TOP_K = 3
MAX_TEXT_LENGTH = 512
MAX_QUERY_LENGTH = 256

BATCH_SIZE = 8

# scoring weights (balanced for research + production)
ALPHA = 0.65   # reranker weight
BETA = 0.25    # vector similarity
GAMMA = 0.10   # hybrid contribution

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# GLOBAL MODEL (LAZY LOAD)
# ─────────────────────────────────────────────
_model = None


def _get_model():
    global _model

    if _model is not None:
        return _model

    try:
        logger.info("[RERANKER] Loading model...")
        _model = CrossEncoder(MODEL_NAME, device=DEVICE)
        logger.info(f"[RERANKER] Loaded on {DEVICE}")
    except Exception as e:
        logger.error(f"[RERANKER] Load failed: {e}")
        _model = None

    return _model


# ─────────────────────────────────────────────
# TEXT SAFETY
# ─────────────────────────────────────────────
def _truncate(text: str, max_len: int) -> str:
    if not text:
        return ""
    return text[:max_len]


# ─────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────
def _normalize(scores: List[float]) -> List[float]:
    if not scores:
        return scores

    min_s, max_s = min(scores), max(scores)

    if max_s == min_s:
        return [0.5] * len(scores)

    return [(s - min_s) / (max_s - min_s) for s in scores]


# ─────────────────────────────────────────────
# FALLBACK RANKING (IMPROVED)
# ─────────────────────────────────────────────
def _fallback_rank(docs: List[Dict[str, Any]], top_k: int):
    logger.warning("[RERANKER] fallback ranking (model unavailable)")

    for d in docs:
        sim = d.get("similarity", 0.5)
        hybrid = d.get("hybrid_score", sim)

        # better fallback blend
        score = 0.7 * sim + 0.3 * hybrid

        d["rerank_score"] = float(sim)
        d["final_score"] = float(score)

    docs.sort(key=lambda x: x["final_score"], reverse=True)

    return docs[:top_k]


# ─────────────────────────────────────────────
# RERANK
# ─────────────────────────────────────────────
def rerank(
    query: str,
    docs: List[Dict[str, Any]],
    top_k: int = TOP_K,
    return_debug: bool = False  # 🔥 NEW (for research)
) -> List[Dict[str, Any]]:

    if not docs:
        return []

    model = _get_model()

    if model is None:
        return _fallback_rank(docs, top_k)

    start = time.time()

    try:
        # ── Safety trims
        query = _truncate(query, MAX_QUERY_LENGTH)
        docs = docs[:MAX_DOCS]

        pairs = []
        valid_docs = []

        for d in docs:
            content = _truncate(d.get("content", ""), MAX_TEXT_LENGTH)

            if not content.strip():
                continue

            pairs.append((query, content))
            valid_docs.append(d)

        if not pairs:
            return []

        # ── Predict (safe execution)
        try:
            raw_scores = model.predict(
                pairs,
                batch_size=BATCH_SIZE,
                show_progress_bar=False
            )
        except RuntimeError as e:
            logger.error(f"[RERANKER] OOM or runtime error: {e}")
            return _fallback_rank(docs, top_k)

        raw_scores = list(raw_scores)
        norm_scores = _normalize(raw_scores)

        # ── Attach scores
        for d, r_score, raw in zip(valid_docs, norm_scores, raw_scores):
            sim = d.get("similarity", 0.0)
            hybrid = d.get("hybrid_score", sim)

            final_score = (
                ALPHA * r_score +
                BETA * sim +
                GAMMA * hybrid
            )

            d.update({
                "rerank_score": float(r_score),
                "cross_score_raw": float(raw),   # 🔥 for research
                "final_score": float(final_score)
            })

        # ── Stable sort
        valid_docs.sort(
            key=lambda x: (x["final_score"], x.get("similarity", 0)),
            reverse=True
        )

        latency = (time.time() - start) * 1000

        logger.info(
            f"[RERANKER] docs={len(valid_docs)} | latency={latency:.2f}ms"
        )

        # 🔥 Debug mode (for analysis)
        if return_debug:
            return {
                "results": valid_docs[:top_k],
                "latency_ms": round(latency, 2),
                "raw_scores": raw_scores
            }

        return valid_docs[:top_k]

    except Exception as e:
        logger.error(f"[RERANKER] error: {e}")
        return _fallback_rank(docs, top_k)