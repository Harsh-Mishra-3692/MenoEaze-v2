# ml_engine/research/uncertainty.py — ELITE v3 (RESEARCH + PRODUCTION)

import logging
import numpy as np
from typing import List, Tuple, Dict

logger = logging.getLogger("menoeaze.uncertainty")

EPS = 1e-8


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────
def _to_array(values: List[float]) -> np.ndarray:
    if not values:
        return np.array([])

    arr = np.array(values, dtype=np.float32)

    # remove NaNs / inf
    arr = arr[np.isfinite(arr)]

    return arr


# ─────────────────────────────────────────────
# VARIANCE UNCERTAINTY
# ─────────────────────────────────────────────
def variance_uncertainty(predictions: List[float]) -> float:
    arr = _to_array(predictions)

    if len(arr) == 0:
        return 0.0

    return float(np.var(arr))


# ─────────────────────────────────────────────
# ENTROPY UNCERTAINTY (NORMALIZED)
# ─────────────────────────────────────────────
def entropy_uncertainty(probs: List[float]) -> float:
    arr = _to_array(probs)

    if len(arr) == 0:
        return 0.0

    # normalize to probability distribution
    total = np.sum(arr) + EPS
    arr = arr / total

    # clamp to avoid log(0)
    arr = np.clip(arr, EPS, 1.0)

    entropy = -np.sum(arr * np.log(arr))

    # normalize entropy to [0,1]
    max_entropy = np.log(len(arr)) if len(arr) > 1 else 1.0

    return float(entropy / (max_entropy + EPS))


# ─────────────────────────────────────────────
# CONFIDENCE INTERVAL (CLAMPED)
# ─────────────────────────────────────────────
def confidence_interval(predictions: List[float], z: float = 1.96) -> Tuple[float, float]:
    arr = _to_array(predictions)

    if len(arr) == 0:
        return (0.0, 0.0)

    mean = float(np.mean(arr))
    std = float(np.std(arr))

    lower = mean - z * std
    upper = mean + z * std

    # clamp to valid prediction range [0,1]
    return (
        float(max(0.0, lower)),
        float(min(1.0, upper))
    )


# ─────────────────────────────────────────────
# CALIBRATED CONFIDENCE SCORE (NEW)
# ─────────────────────────────────────────────
def confidence_score(predictions: List[float]) -> float:
    """
    Converts variance → confidence
    lower variance = higher confidence
    """
    var = variance_uncertainty(predictions)

    # inverse relationship
    confidence = 1.0 / (1.0 + var)

    return float(np.clip(confidence, 0.0, 1.0))


# ─────────────────────────────────────────────
# UNCERTAINTY SUMMARY (FOR REPORT)
# ─────────────────────────────────────────────
def uncertainty_summary(predictions: List[float]) -> Dict[str, float]:
    arr = _to_array(predictions)

    if len(arr) == 0:
        return {
            "mean": 0.0,
            "variance": 0.0,
            "confidence": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
        }

    mean = float(np.mean(arr))
    var = variance_uncertainty(arr.tolist())
    conf = confidence_score(arr.tolist())
    ci_low, ci_high = confidence_interval(arr.tolist())

    return {
        "mean": mean,
        "variance": var,
        "confidence": conf,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
    }