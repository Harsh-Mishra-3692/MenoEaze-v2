# personalization_trainer.py — FINAL HARDENED (STABLE + SAFE + DRIFT-CONTROLLED)

import logging
from typing import Dict, Any, List, Optional
import numpy as np

from ml_engine.db_client import (
    fetch_feedback_history,
    fetch_recent_logs,
    get_user_weights,
    update_user_weights
)
from ml_engine.trust_filter import compute_trust_single

logger = logging.getLogger("menoeaze.personalization_trainer")

FEATURES = 11
MAX_SAMPLES = 50
MIN_SAMPLES = 5

TRUST_THRESHOLD = 0.25
OUTLIER_ERR = 0.8

LR_BIAS = 0.02
LR_FEATURE = 0.01
LR_LORA = 0.01
LR_TEMPORAL = 0.01

MAX_GLOBAL_ADJUST = 0.15
MAX_FEATURE_ADJUST = 0.10
MAX_TEMPORAL_ADJUST = 0.08

EMA_ALPHA = 0.2
TIME_DECAY = 0.95

STABILITY_SOFT = 0.05
DRIFT_LIMIT = 0.25

MAX_CONFIDENCE = 0.95
CONFIDENCE_DECAY = 0.02


# ─────────────────────────────────────────────
# SAFE HELPERS
# ─────────────────────────────────────────────
def _safe_vec(v):
    try:
        arr = np.array(v, dtype=np.float32)
        if arr.shape[0] == FEATURES and np.isfinite(arr).all():
            return arr
    except:
        pass
    return np.zeros(FEATURES, dtype=np.float32)


def _clamp(v, lo=-1.0, hi=1.0):
    try:
        return max(lo, min(hi, float(v)))
    except:
        return 0.0


def _clamp01(v):
    try:
        return max(0.0, min(1.0, float(v)))
    except:
        return 0.0


def _ema(old, new):
    return (1 - EMA_ALPHA) * old + EMA_ALPHA * new


# ─────────────────────────────────────────────
# FAST TIME ALIGNMENT (O(n log n))
# ─────────────────────────────────────────────
def _align_by_time(feedback, logs):

    logs_sorted = sorted(logs, key=lambda x: float(x.get("created_at", 0)))

    aligned = []

    for fb in feedback:
        fb_ts = float(fb.get("created_at", 0))

        # binary search
        lo, hi = 0, len(logs_sorted) - 1
        best = None
        best_diff = float("inf")

        while lo <= hi:
            mid = (lo + hi) // 2
            lg = logs_sorted[mid]

            diff = abs(float(lg.get("created_at", 0)) - fb_ts)

            if diff < best_diff:
                best_diff = diff
                best = lg

            if float(lg.get("created_at", 0)) < fb_ts:
                lo = mid + 1
            else:
                hi = mid - 1

        # only accept close matches
        if best and best_diff < 3600:  # 1 hour tolerance
            aligned.append((fb, best))

    return aligned


# ─────────────────────────────────────────────
# PREP DATA (STABLE)
# ─────────────────────────────────────────────
def _prepare_data(feedback, logs):

    pairs = _align_by_time(feedback[-MAX_SAMPLES:], logs[-MAX_SAMPLES:])

    X, E, W = [], [], []

    for i, (f, l) in enumerate(pairs):

        vec = l.get("feature_vector")
        if not isinstance(vec, list) or len(vec) != FEATURES:
            continue

        try:
            pred = float(f.get("predicted", 0.5))
            act = float(f.get("actual", 0.5))

            if not (0 <= pred <= 1 and 0 <= act <= 1):
                continue

            err = act - pred

            if abs(err) > OUTLIER_ERR:
                continue

            trust = compute_trust_single(f)
            if trust < TRUST_THRESHOLD:
                continue

            decay = TIME_DECAY ** (len(pairs) - i)
            weight = trust * decay

            X.append(vec)
            E.append(err)
            W.append(weight)

        except:
            continue

    if not X:
        return None, None, None

    X = np.array(X, dtype=np.float32)

    # safe normalization
    std = np.std(X, axis=0) + 1e-6
    X = np.clip(X / std, -5, 5)

    W = np.array(W, dtype=np.float32)
    W = W / (np.sum(W) + 1e-6)

    return X, np.array(E), W


# ─────────────────────────────────────────────
# COMPUTE UPDATES (SOFT STABILITY)
# ─────────────────────────────────────────────
def _compute_updates(X, e, w):

    if len(X) < MIN_SAMPLES:
        return None

    weighted_e = e * w

    std = np.std(weighted_e)
    stability_factor = np.exp(-std / (STABILITY_SOFT + 1e-6))

    bias = np.mean(weighted_e) * LR_BIAS * stability_factor

    corr = (X * weighted_e[:, None]).mean(axis=0)

    importance = np.abs(corr)
    mask = importance > np.percentile(importance, 60)
    corr *= mask

    norm = np.linalg.norm(corr) + 1e-6
    feature = (corr / norm) * LR_FEATURE * stability_factor

    var = X.var(axis=0) + 1e-6
    A = (corr / norm) * LR_LORA
    B = (var / (np.linalg.norm(var) + 1e-6)) * LR_LORA

    slope = e[-1] - e[0] if len(e) > 2 else 0
    temporal = slope * LR_TEMPORAL

    return {
        "bias": _clamp(bias, -MAX_GLOBAL_ADJUST, MAX_GLOBAL_ADJUST),
        "feature_weights": np.clip(feature, -MAX_FEATURE_ADJUST, MAX_FEATURE_ADJUST),
        "lora_A": np.clip(A, -MAX_FEATURE_ADJUST, MAX_FEATURE_ADJUST),
        "lora_B": np.clip(B, -MAX_FEATURE_ADJUST, MAX_FEATURE_ADJUST),
        "temporal_weight": _clamp(temporal, -MAX_TEMPORAL_ADJUST, MAX_TEMPORAL_ADJUST)
    }


# ─────────────────────────────────────────────
# MERGE WITH GLOBAL DRIFT CONTROL
# ─────────────────────────────────────────────
def _merge(existing, updates):

    existing = existing or {}

    bias = _ema(existing.get("bias", 0.0), updates["bias"])
    bias = _clamp(bias, -DRIFT_LIMIT, DRIFT_LIMIT)

    fw = _ema(_safe_vec(existing.get("feature_weights")), _safe_vec(updates["feature_weights"]))
    fw = np.clip(fw, -MAX_FEATURE_ADJUST, MAX_FEATURE_ADJUST)

    A = _ema(_safe_vec(existing.get("lora_A")), _safe_vec(updates["lora_A"]))
    B = _ema(_safe_vec(existing.get("lora_B")), _safe_vec(updates["lora_B"]))

    temp = _ema(existing.get("temporal_weight", 0.0), updates["temporal_weight"])
    temp = _clamp(temp, -MAX_TEMPORAL_ADJUST, MAX_TEMPORAL_ADJUST)

    # confidence logic (stable)
    confidence = existing.get("confidence", 0.0)

    if abs(updates["bias"]) > 0.01:
        confidence += 0.04
    else:
        confidence -= CONFIDENCE_DECAY

    confidence = _clamp01(min(confidence, MAX_CONFIDENCE))

    return {
        "bias": float(bias),
        "feature_weights": fw.tolist(),
        "lora_A": A.tolist(),
        "lora_B": B.tolist(),
        "temporal_weight": float(temp),
        "confidence": float(confidence)
    }


# ─────────────────────────────────────────────
# MAIN ENTRY
# ─────────────────────────────────────────────
def update_personalization_from_feedback(user_id: str) -> bool:

    if not user_id:
        return False

    try:
        feedback = fetch_feedback_history(user_id, limit=MAX_SAMPLES) or []
        logs = fetch_recent_logs(user_id, limit=MAX_SAMPLES) or []

        if not feedback or not logs:
            return False

        X, e, w = _prepare_data(feedback, logs)

        if X is None:
            return False

        updates = _compute_updates(X, e, w)

        if updates is None:
            return False

        existing = get_user_weights(user_id)
        merged = _merge(existing, updates)

        ok = update_user_weights(user_id, merged)

        logger.info(
            f"[Trainer] user={user_id} "
            f"bias={round(merged['bias'],4)} "
            f"conf={round(merged['confidence'],4)}"
        )

        return bool(ok)

    except Exception as e:
        logger.error(f"[Trainer] failed: {e}")
        return False