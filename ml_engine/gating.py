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
        'none' | 'bias' | 'maml' | 'adapt'
    """

    try:
        if not user_history:
            return "none"

        predictions = user_history.get("predictions", [])
        actuals = user_history.get("actuals", [])
        feedback_logs = user_history.get("feedback_logs", [])

        # Phase 5: MAML uses feedback_logs (raw sequence + actual_severity dicts)
        # independently of the predictions/actuals alignment.
        # Check this FIRST since a user may have feedback_logs without
        # aligned pred/actual pairs (e.g., submitted feedback but no /run calls).
        n_feedback = len(feedback_logs)

        # Standard pred/actual alignment count
        n = min(len(predictions), len(actuals)) if predictions and actuals else 0

        # ── Not enough data from either source ───
        if n < MIN_HISTORY and n_feedback < MAML_THRESHOLD:
            return "none"

        # ── Extract features (only if we have aligned pred/actual data) ──
        if n >= MIN_HISTORY:
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
        # Requires feedback_logs with sequence + actual_severity.
        # Fewer samples needed than full adapt, more than bias.
        if n_feedback >= MAML_THRESHOLD:
            return "maml"

        # ── Have some pred/actual data but not enough for MAML ───
        if n >= MIN_HISTORY:
            return "bias"

        # ── Default safe option ───────────────
        return "none"

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
