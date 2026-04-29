# bias_control.py — ELITE (STABLE + INTELLIGENT BIAS CONTROL)

import logging
import numpy as np
from typing import List, Dict

logger = logging.getLogger("menoeaze.bias")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MAX_HISTORY = 50
OUTLIER_Z = 2.5
BIAS_CLAMP = 0.2
MIN_SAMPLES = 5

RECENCY_ALPHA = 0.6
VARIANCE_DAMPING = 0.5

EMA_ALPHA = 0.2              # NEW: temporal smoothing
MIN_CONF_SAMPLES = 10        # NEW: confidence scaling
TREND_WEIGHT = 0.1           # NEW: directional correction


# ─────────────────────────────────────────────
# OUTLIER REMOVAL
# ─────────────────────────────────────────────
def _remove_outliers(errors: np.ndarray) -> np.ndarray:
    if len(errors) < 5:
        return errors

    mean = np.mean(errors)
    std = np.std(errors)

    if std == 0 or np.isnan(std):
        return errors

    z = np.abs((errors - mean) / std)
    return errors[z < OUTLIER_Z]


# ─────────────────────────────────────────────
# TRIMMED MEAN
# ─────────────────────────────────────────────
def _trimmed_mean(errors: np.ndarray, trim_ratio=0.1) -> float:
    if len(errors) < 5:
        return float(np.mean(errors))

    sorted_err = np.sort(errors)
    n = len(sorted_err)
    k = int(n * trim_ratio)

    trimmed = sorted_err[k:n - k] if n > 2 * k else sorted_err
    return float(np.mean(trimmed))


# ─────────────────────────────────────────────
# RECENCY WEIGHTING
# ─────────────────────────────────────────────
def _recency_weighted(errors: np.ndarray) -> float:
    if len(errors) == 0:
        return 0.0

    weights = np.linspace(0.5, 1.0, len(errors))
    weights /= weights.sum()

    return float(np.sum(errors * weights))


# ─────────────────────────────────────────────
# TREND DETECTION
# ─────────────────────────────────────────────
def _trend(errors: np.ndarray) -> float:
    if len(errors) < 3:
        return 0.0
    return float(errors[-1] - errors[0])


# ─────────────────────────────────────────────
# VARIANCE DAMPING
# ─────────────────────────────────────────────
def _variance_penalty(errors: np.ndarray) -> float:
    if len(errors) < 2:
        return 1.0

    var = np.var(errors)
    penalty = 1 / (1 + VARIANCE_DAMPING * var)

    return float(np.clip(penalty, 0.3, 1.0))


# ─────────────────────────────────────────────
# SAMPLE-SIZE CONFIDENCE
# ─────────────────────────────────────────────
def _sample_confidence(n: int) -> float:
    return float(np.clip(n / MIN_CONF_SAMPLES, 0.0, 1.0))


# ─────────────────────────────────────────────
# EMA SMOOTHING
# ─────────────────────────────────────────────
def _ema(errors: np.ndarray) -> float:
    ema = 0.0
    for e in errors:
        ema = EMA_ALPHA * e + (1 - EMA_ALPHA) * ema
    return float(ema)


# ─────────────────────────────────────────────
# MAIN BIAS COMPUTATION
# ─────────────────────────────────────────────
def compute_stable_bias(
    predictions: List[float],
    actuals: List[float]
) -> Dict:

    if (
        not predictions
        or not actuals
        or len(predictions) != len(actuals)
        or len(predictions) < MIN_SAMPLES
    ):
        return {
            "bias": 0.0,
            "confidence": 0.0,
            "variance": 0.0,
            "n": 0
        }

    try:
        errors = np.array([a - p for a, p in zip(actuals, predictions)])[-MAX_HISTORY:]

        # ── Outlier removal ───────────────────
        filtered = _remove_outliers(errors)
        n = len(filtered)

        if n == 0:
            return {"bias": 0.0, "confidence": 0.0, "variance": 0.0, "n": 0}

        # ── Core signals ──────────────────────
        robust = _trimmed_mean(filtered)
        recency = _recency_weighted(filtered)
        ema_val = _ema(filtered)
        trend = _trend(filtered)

        # ── Combine signals ───────────────────
        base_bias = (
            0.4 * robust +
            0.4 * recency +
            0.2 * ema_val
        )

        # ── Trend adjustment ──────────────────
        base_bias += TREND_WEIGHT * trend

        # ── Variance damping ──────────────────
        penalty = _variance_penalty(filtered)
        bias = base_bias * penalty

        # ── Clamp ─────────────────────────────
        bias = float(np.clip(bias, -BIAS_CLAMP, BIAS_CLAMP))

        # ── Confidence ────────────────────────
        variance = float(np.var(filtered))
        var_conf = np.clip(1 - variance, 0.0, 1.0)
        size_conf = _sample_confidence(n)

        confidence = float(np.clip(0.5 * var_conf + 0.5 * size_conf, 0.0, 1.0))

        return {
            "bias": round(bias, 4),
            "confidence": round(confidence, 3),
            "variance": round(variance, 4),
            "trend": round(trend, 4),
            "n": n
        }

    except Exception as e:
        logger.error(f"[Bias] Failed: {e}")
        return {
            "bias": 0.0,
            "confidence": 0.0,
            "variance": 0.0,
            "n": 0
        }


# ─────────────────────────────────────────────
# APPLY FINAL BIAS SAFELY
# ─────────────────────────────────────────────
def apply_bias(
    prediction: float,
    bias: float,
    confidence: float
) -> float:

    try:
        # scale bias conservatively
        scaled_bias = bias * confidence

        # hard safety: limit total adjustment
        max_adjust = 0.25
        scaled_bias = float(np.clip(scaled_bias, -max_adjust, max_adjust))

        adjusted = prediction + scaled_bias

        # final clamp
        adjusted = float(np.clip(adjusted, 0.0, 1.0))

        if np.isnan(adjusted):
            return prediction

        return round(adjusted, 4)

    except Exception:
        return prediction
