# gating.py — ELITE (PERSONALIZATION STRATEGY SELECTOR)

import logging
from typing import Dict, Optional, List

import numpy as np

logger = logging.getLogger("menoeaze.gating")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MIN_HISTORY = 5
MAML_THRESHOLD = 5      # minimum feedback logs for MAML fast adaptation
ADAPT_THRESHOLD = 15

MAX_VARIANCE = 0.25     # noisy data cutoff
TREND_THRESHOLD = 0.05  # signal strength threshold


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────
def _extract_error_features(
    predictions: List[float],
    actuals: List[float]
) -> Dict:

    if not predictions or not actuals:
        return {
            "variance": 1.0,
            "trend": 0.0,
            "mean_error": 0.0
        }

    errors = np.array(actuals) - np.array(predictions)

    variance = float(np.var(errors))

    trend = float(errors[-1] - errors[0]) if len(errors) > 2 else 0.0

    mean_error = float(np.mean(errors))

    return {
        "variance": variance,
        "trend": trend,
        "mean_error": mean_error
    }


# ─────────────────────────────────────────────
# MAIN GATING LOGIC
# ─────────────────────────────────────────────
def select_strategy(
    user_history: Optional[Dict]
) -> str:
    """
    Returns:
        'none' | 'bias' | 'adapt'
    """

    try:
        if not user_history:
            return "none"

        predictions = user_history.get("predictions", [])
        actuals = user_history.get("actuals", [])

        if not predictions or not actuals:
            return "none"

        n = min(len(predictions), len(actuals))

        # ── Not enough data ───────────────────
        if n < MIN_HISTORY:
            return "none"

        # ── Extract features ──────────────────
        feats = _extract_error_features(predictions[-30:], actuals[-30:])

        variance = feats["variance"]
        trend = abs(feats["trend"])

        # ── Noisy data → avoid adaptation ─────
        if variance > MAX_VARIANCE:
            return "bias"

        # ── Enough data for full adaptation ───
        if n >= ADAPT_THRESHOLD and trend > TREND_THRESHOLD:
            return "adapt"

        # ── Phase 5: MAML fast adaptation ─────
        # Requires fewer samples than full adapt but more than bias.
        # Works alongside EMA — only triggers if user has real feedback.
        if n >= MAML_THRESHOLD:
            return "maml"

        # ── Default safe option ───────────────
        return "bias"

    except Exception as e:
        logger.error(f"[Gating] failed: {e}")
        return "none"


# ─────────────────────────────────────────────
# DEBUG INFO (OPTIONAL)
# ─────────────────────────────────────────────
def explain_strategy(
    user_history: Optional[Dict]
) -> Dict:
    """
    Returns detailed reasoning for debugging / analytics
    """

    if not user_history:
        return {"strategy": "none", "reason": "no_history"}

    predictions = user_history.get("predictions", [])
    actuals = user_history.get("actuals", [])

    n = min(len(predictions), len(actuals))

    if n < MIN_HISTORY:
        return {"strategy": "none", "reason": "insufficient_data"}

    feats = _extract_error_features(predictions[-30:], actuals[-30:])

    if feats["variance"] > MAX_VARIANCE:
        return {
            "strategy": "bias",
            "reason": "high_variance",
            "features": feats
        }

    if n >= ADAPT_THRESHOLD and abs(feats["trend"]) > TREND_THRESHOLD:
        return {
            "strategy": "adapt",
            "reason": "stable_signal",
            "features": feats
        }

    return {
        "strategy": "bias",
        "reason": "default_safe",
        "features": feats
    }
