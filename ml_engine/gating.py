# gating.py — ELITE PERSONALIZATION GATING (MENOEAZE)

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np

logger = logging.getLogger("menoeaze.gating")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MIN_HISTORY = 5
MAX_VARIANCE = 0.20
TREND_THRESHOLD = 0.04

MAX_FEEDBACK_AGE_DAYS = 30
MIN_TRUST_SCORE = 0.4

MIN_USER_RELIABILITY = 0.45
MIN_CONFIDENCE_REQUIRED = 0.4

OSCILLATION_THRESHOLD = 0.25


# ─────────────────────────────────────────────
# UTIL
# ─────────────────────────────────────────────
def _safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default


def _days_diff(ts: float) -> float:
    try:
        return (datetime.utcnow().timestamp() - float(ts)) / 86400.0
    except:
        return 999.0


# ─────────────────────────────────────────────
# FILTER FEEDBACK
# ─────────────────────────────────────────────
def _filter_feedback(feedback: List[Dict]) -> List[Dict]:

    filtered = []

    for f in feedback:
        try:
            trust = _safe_float(f.get("trust_score", 0))
            ts = _safe_float(f.get("ts", 0))

            if trust < MIN_TRUST_SCORE:
                continue

            if _days_diff(ts) > MAX_FEEDBACK_AGE_DAYS:
                continue

            filtered.append(f)

        except:
            continue

    return filtered


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────
def _extract_features(preds: List[float], actuals: List[float]):

    errors = np.array(actuals) - np.array(preds)

    variance = float(np.var(errors))
    trend = float(errors[-1] - errors[0]) if len(errors) > 2 else 0.0

    # oscillation = instability
    diffs = np.diff(errors) if len(errors) > 2 else []
    oscillation = float(np.mean(np.abs(diffs))) if len(diffs) else 0.0

    return variance, trend, oscillation


# ─────────────────────────────────────────────
# MAIN GATING
# ─────────────────────────────────────────────
def select_strategy(
    history: Optional[List[Dict]],
    feedback: Optional[List[Dict]],
    model_confidence: float = 0.5,
    user_reliability: float = 0.0
) -> str:
    """
    Returns:
        'none' | 'bias'
    """

    try:
        history = history or []
        feedback = feedback or []

        # ── basic guards ───────────────────
        if model_confidence < MIN_CONFIDENCE_REQUIRED:
            return "none"

        if user_reliability < MIN_USER_RELIABILITY:
            return "none"

        # ── filter feedback ────────────────
        feedback = _filter_feedback(feedback)

        preds = []
        actuals = []

        for f in feedback:
            preds.append(_safe_float(f.get("predicted", 0.5)))
            actuals.append(_safe_float(f.get("actual", 0.5)))

        n = min(len(preds), len(actuals))

        if n < MIN_HISTORY:
            return "none"

        variance, trend, oscillation = _extract_features(
            preds[-30:], actuals[-30:]
        )

        # ── noisy → disable ────────────────
        if variance > MAX_VARIANCE:
            return "none"

        # ── unstable user → disable ────────
        if oscillation > OSCILLATION_THRESHOLD:
            return "none"

        # ── stable signal → allow bias ─────
        if abs(trend) > TREND_THRESHOLD:
            return "bias"

        return "none"

    except Exception as e:
        logger.error(f"[Gating] failed: {e}")
        return "none"


# ─────────────────────────────────────────────
# EXPLAINABILITY
# ─────────────────────────────────────────────
def explain_strategy(
    history: Optional[List[Dict]],
    feedback: Optional[List[Dict]],
    model_confidence: float = 0.5,
    user_reliability: float = 0.0
) -> Dict[str, Any]:

    try:
        feedback = _filter_feedback(feedback or [])

        preds = []
        actuals = []

        for f in feedback:
            preds.append(_safe_float(f.get("predicted", 0.5)))
            actuals.append(_safe_float(f.get("actual", 0.5)))

        n = min(len(preds), len(actuals))

        if model_confidence < MIN_CONFIDENCE_REQUIRED:
            return {"strategy": "none", "reason": "low_model_confidence"}

        if user_reliability < MIN_USER_RELIABILITY:
            return {"strategy": "none", "reason": "low_user_reliability"}

        if n < MIN_HISTORY:
            return {"strategy": "none", "reason": "insufficient_data"}

        variance, trend, oscillation = _extract_features(
            preds[-30:], actuals[-30:]
        )

        if variance > MAX_VARIANCE:
            return {
                "strategy": "none",
                "reason": "high_variance",
                "variance": variance
            }

        if oscillation > OSCILLATION_THRESHOLD:
            return {
                "strategy": "none",
                "reason": "unstable_feedback",
                "oscillation": oscillation
            }

        if abs(trend) > TREND_THRESHOLD:
            return {
                "strategy": "bias",
                "reason": "stable_trend",
                "trend": trend
            }

        return {
            "strategy": "none",
            "reason": "weak_signal",
            "trend": trend
        }

    except Exception as e:
        return {
            "strategy": "none",
            "reason": "error",
            "error": str(e)
        }