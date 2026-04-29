# ml_engine/research/semantic_metrics.py — ELITE v3 (RESEARCH + PRODUCTION)

import logging
import time
import numpy as np
from typing import List, Dict

from sentence_transformers import SentenceTransformer

logger = logging.getLogger("menoeaze.semantic")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32

_model = None


# ─────────────────────────────────────────────
# MODEL LOADER (LAZY + SAFE)
# ─────────────────────────────────────────────
def _get_model():
    global _model
    if _model is None:
        try:
            logger.info("[Semantic] Loading embedding model...")
            _model = SentenceTransformer(MODEL_NAME)
            logger.info("[Semantic] Model ready")
        except Exception as e:
            logger.error(f"[Semantic] Model load failed: {e}")
            raise
    return _model


# ─────────────────────────────────────────────
# SAFE COSINE SIMILARITY
# ─────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


# ─────────────────────────────────────────────
# SINGLE SIMILARITY
# ─────────────────────────────────────────────
def semantic_similarity(text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 0.0

    try:
        model = _get_model()

        emb = model.encode(
            [text_a, text_b],
            normalize_embeddings=True,
            batch_size=2
        )

        return float(np.dot(emb[0], emb[1]))

    except Exception as e:
        logger.error(f"[Semantic] similarity failed: {e}")
        return 0.0


# ─────────────────────────────────────────────
# BATCH SIMILARITY (OPTIMIZED)
# ─────────────────────────────────────────────
def batch_similarity(pairs: List[Dict[str, str]]) -> List[float]:
    if not pairs:
        return []

    try:
        model = _get_model()

        texts = []
        for p in pairs:
            a = p.get("a", "")
            b = p.get("b", "")
            texts.extend([a, b])

        start = time.time()

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=BATCH_SIZE
        )

        results = []
        for i in range(0, len(embeddings), 2):
            sim = float(np.dot(embeddings[i], embeddings[i + 1]))
            results.append(sim)

        latency = (time.time() - start) * 1000

        logger.info(
            f"[Semantic] batch_size={len(pairs)} | latency={latency:.2f}ms"
        )

        return results

    except Exception as e:
        logger.error(f"[Semantic] batch failed: {e}")
        return [0.0] * len(pairs)


# ─────────────────────────────────────────────
# ADVANCED METRIC (NEW)
# ─────────────────────────────────────────────
def semantic_stats(scores: List[float]) -> Dict[str, float]:
    """
    Useful for research/report section
    """
    if not scores:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    arr = np.array(scores)

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }