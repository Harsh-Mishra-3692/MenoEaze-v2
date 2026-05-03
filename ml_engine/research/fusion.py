# fusion.py — FINAL ELITE v3 (STABLE + CALIBRATED + ROBUST)

from typing import List, Dict
import math
import hashlib

DEFAULT_ALPHA = 0.6
EPSILON = 1e-9


# ─────────────────────────────────────────────
# SAFE FLOAT
# ─────────────────────────────────────────────
def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


# ─────────────────────────────────────────────
# ROBUST NORMALIZATION (Z-SCORE + MINMAX)
# ─────────────────────────────────────────────
def _normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []

    # remove NaNs
    scores = [_safe_float(s) for s in scores]

    mean = sum(scores) / (len(scores) + EPSILON)
    var = sum((s - mean) ** 2 for s in scores) / (len(scores) + EPSILON)
    std = math.sqrt(var + EPSILON)

    # fallback if low variance
    if std < 1e-6:
        mn, mx = min(scores), max(scores)
        if abs(mx - mn) < EPSILON:
            return [1.0 for _ in scores]
        return [(s - mn) / (mx - mn + EPSILON) for s in scores]

    # z-score normalize → sigmoid squash
    normalized = []
    for s in scores:
        z = (s - mean) / std
        val = 1 / (1 + math.exp(-z))  # sigmoid
        normalized.append(val)

    return normalized


# ─────────────────────────────────────────────
# STABLE HASH (DEDUP)
# ─────────────────────────────────────────────
def _doc_hash(d: Dict, key: str = "content") -> str:
    text = str(d.get(key, ""))[:500]
    return hashlib.sha256(text.encode()).hexdigest()


# ─────────────────────────────────────────────
# DEDUPLICATION
# ─────────────────────────────────────────────
def _deduplicate(docs: List[Dict]) -> Dict[str, Dict]:
    merged = {}

    for d in docs:
        h = _doc_hash(d)

        if h not in merged:
            merged[h] = d.copy()
        else:
            existing = merged[h]

            # merge best scores
            existing["score"] = max(
                _safe_float(existing.get("score")),
                _safe_float(d.get("score"))
            )

            existing["bm25_score"] = max(
                _safe_float(existing.get("bm25_score")),
                _safe_float(d.get("bm25_score"))
            )

    return merged


# ─────────────────────────────────────────────
# SCORE CALIBRATION
# ─────────────────────────────────────────────
def _calibrate(v: float, b: float) -> tuple:
    """
    Balance vector vs bm25 dominance
    """
    v = _safe_float(v)
    b = _safe_float(b)

    # prevent one signal dominating completely
    if v > 0 and b == 0:
        b = v * 0.3
    elif b > 0 and v == 0:
        v = b * 0.3

    return v, b


# ─────────────────────────────────────────────
# MAIN FUSION
# ─────────────────────────────────────────────
def fuse_results(
    vector_docs: List[Dict],
    bm25_docs: List[Dict],
    alpha: float = DEFAULT_ALPHA,
    top_k: int = 10
) -> List[Dict]:

    if not vector_docs and not bm25_docs:
        return []

    alpha = max(0.0, min(1.0, alpha))

    combined = (vector_docs or []) + (bm25_docs or [])
    merged = _deduplicate(combined)

    docs = list(merged.values())

    if not docs:
        return []

    vector_scores = [_safe_float(d.get("score", 0.0)) for d in docs]
    bm25_scores = [_safe_float(d.get("bm25_score", 0.0)) for d in docs]

    vector_norm = _normalize(vector_scores)
    bm25_norm = _normalize(bm25_scores)

    for i, d in enumerate(docs):
        v = vector_norm[i]
        b = bm25_norm[i]

        # calibration step
        v, b = _calibrate(v, b)

        fused = alpha * v + (1 - alpha) * b

        # prevent pathological scores
        fused = max(0.0, min(1.0, fused))

        d["fusion_score"] = float(fused)

        # 🔍 explainability (optional but powerful)
        d["_fusion_meta"] = {
            "vector": round(v, 4),
            "bm25": round(b, 4),
            "alpha": alpha
        }

    docs.sort(key=lambda x: x["fusion_score"], reverse=True)

    return docs[:top_k]