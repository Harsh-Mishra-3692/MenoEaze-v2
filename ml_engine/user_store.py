# user_store.py — FINAL ELITE v2 (TEMPORAL-ALIGNED + SIGNAL-AWARE)

import logging
from typing import List, Dict, Tuple
from datetime import datetime

import numpy as np

from ml_engine.db_client import (
    fetch_prediction_history,
    fetch_feedback_history,
    fetch_recent_logs
)

logger = logging.getLogger("menoeaze.user_store")

SEQ_LEN = 5
FEATURES = 11
MAX_SAMPLES = 50
MIN_TRUST = 0.3
OUTLIER_THRESHOLD = 0.8
MAX_TIME_DIFF_SEC = 3600

EPS = 1e-6


# ─────────────────────────────────────────────
# SAFE HELPERS
# ─────────────────────────────────────────────
def _safe_float(x, default=0.0):
    try:
        x = float(x)
        return x if np.isfinite(x) else default
    except:
        return default


def _clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def _parse_ts(ts):
    try:
        if isinstance(ts, (int, float)):
            return float(ts)
        return datetime.fromisoformat(str(ts)).timestamp()
    except:
        return None


def _valid_vector(vec):
    return (
        isinstance(vec, list)
        and len(vec) == FEATURES
        and all(isinstance(v, (int, float)) for v in vec)
    )


def _valid_sequence(seq: np.ndarray) -> bool:
    if not isinstance(seq, np.ndarray):
        return False
    if seq.shape != (SEQ_LEN, FEATURES):
        return False
    if not np.isfinite(seq).all():
        return False
    if np.mean(seq) == 0:  # all padding
        return False
    return True


# ─────────────────────────────────────────────
# BUILD SEQUENCES
# ─────────────────────────────────────────────
def _build_sequences(logs: List[Dict]):

    cleaned = []

    for l in logs:
        vec = l.get("feature_vector")
        ts = _parse_ts(l.get("created_at"))

        if not _valid_vector(vec) or ts is None:
            continue

        arr = np.array(vec, dtype=np.float32)

        if not np.isfinite(arr).all():
            continue

        cleaned.append((ts, arr))

    if not cleaned:
        return []

    cleaned.sort(key=lambda x: x[0])

    sequences = []

    for i in range(len(cleaned)):
        start = max(0, i - SEQ_LEN + 1)
        seq = [v for _, v in cleaned[start:i + 1]]

        if len(seq) < SEQ_LEN:
            pad = [np.zeros(FEATURES, dtype=np.float32)] * (SEQ_LEN - len(seq))
            seq = pad + seq

        seq = np.stack(seq)

        if _valid_sequence(seq):
            sequences.append((cleaned[i][0], seq))  # attach timestamp

    return sequences


# ─────────────────────────────────────────────
# ALIGN FEEDBACK (TIME-AWARE)
# ─────────────────────────────────────────────
def _align_feedback(preds, feedback):

    pred_map = []

    for p in preds:
        ts = _parse_ts(p.get("created_at"))
        if ts is None:
            continue

        pred_map.append({
            "id": p.get("id"),
            "severity": _safe_float(p.get("severity")),
            "ts": ts
        })

    aligned = []

    for fb in feedback:

        trust = _safe_float(fb.get("trust_score"))
        if trust < MIN_TRUST:
            continue

        actual = _safe_float(fb.get("actual"))
        if not (0 <= actual <= 1):
            continue

        fb_ts = _parse_ts(fb.get("created_at"))
        if fb_ts is None:
            continue

        # ── find closest prediction
        best = None
        best_diff = float("inf")

        for p in pred_map:
            diff = abs(p["ts"] - fb_ts)
            if diff < best_diff:
                best_diff = diff
                best = p

        if not best or best_diff > MAX_TIME_DIFF_SEC:
            continue

        error = abs(best["severity"] - actual)

        if error > OUTLIER_THRESHOLD:
            continue

        aligned.append({
            "pred": best["severity"],
            "actual": actual,
            "error": error,
            "ts": best["ts"],
            "weight": trust
        })

    if not aligned:
        return []

    # normalize weights
    weights = np.array([a["weight"] for a in aligned])
    weights = weights / (np.sum(weights) + EPS)

    for i, a in enumerate(aligned):
        a["weight"] = float(weights[i])

    return aligned


# ─────────────────────────────────────────────
# MAIN HISTORY
# ─────────────────────────────────────────────
def get_user_history(user_id: str):

    try:
        preds = fetch_prediction_history(user_id, limit=MAX_SAMPLES) or []
        feedback = fetch_feedback_history(user_id, limit=MAX_SAMPLES) or []
        logs = fetch_recent_logs(user_id, limit=MAX_SAMPLES) or []

        sequences = _build_sequences(logs)
        aligned = _align_feedback(preds, feedback)

        if not sequences or len(aligned) < 2:
            return None, None, [], []

        X, y = [], []
        pred_vals, actual_vals = [], []

        for item in aligned:

            # ── find closest sequence in time
            best_seq = None
            best_diff = float("inf")

            for ts, seq in sequences:
                diff = abs(ts - item["ts"])
                if diff < best_diff:
                    best_diff = diff
                    best_seq = seq

            if best_seq is None:
                continue

            X.append(best_seq)
            y.append(item["actual"])

            pred_vals.append(item["pred"])
            actual_vals.append(item["actual"])

        if not X:
            return None, None, [], []

        return (
            np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            pred_vals,
            actual_vals
        )

    except Exception as e:
        logger.error(f"[UserStore] history failed: {e}")
        return None, None, [], []


# ─────────────────────────────────────────────
# USER RELIABILITY (UPGRADED)
# ─────────────────────────────────────────────
def compute_user_reliability(user_id: str):

    try:
        feedback = fetch_feedback_history(user_id, limit=20) or []

        if not feedback:
            return 0.0

        scores = [
            _safe_float(fb.get("trust_score"))
            for fb in feedback
            if fb.get("trust_score") is not None
        ]

        if not scores:
            return 0.0

        mean = np.mean(scores)
        variance = np.var(scores)

        stability = np.exp(-variance * 5)

        # signal strength factor
        strength = min(1.0, len(scores) / 20)

        return float(_clamp(mean * stability * strength))

    except Exception:
        return 0.0