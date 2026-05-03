# trust_filter.py — FINAL ELITE v3 (ANTI-POISONING + STABLE + AUDITABLE)

import math
from typing import List, Dict, Optional
from collections import Counter

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
REJECTION_THRESHOLD = 0.6
MIN_HISTORY = 3
DECAY_FACTOR = 0.9

CONSISTENCY_WINDOW = 5
DUPLICATE_TOLERANCE = 0.01
MAX_HISTORY = 30

MIN_VALID_VALUE = 0.0
MAX_VALID_VALUE = 1.0

EPS = 1e-9


# ─────────────────────────────────────────────
# SAFE HELPERS
# ─────────────────────────────────────────────
def _safe_float(x, default=0.5):
    try:
        x = float(x)
        if not (x == x):
            return default
        return x
    except:
        return default


def _clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def _valid_range(x):
    return MIN_VALID_VALUE <= x <= MAX_VALID_VALUE


# ─────────────────────────────────────────────
# CORE TRUST TRANSFORM
# ─────────────────────────────────────────────
def _error_to_trust(error: float) -> float:
    return math.exp(-3 * error)


# ─────────────────────────────────────────────
# SINGLE FEEDBACK TRUST
# ─────────────────────────────────────────────
def compute_trust_single(feedback: Dict) -> float:

    pred = _safe_float(feedback.get("predicted"))
    actual = _safe_float(feedback.get("actual"))

    if not (_valid_range(pred) and _valid_range(actual)):
        return 0.0

    error = abs(pred - actual)

    if error > REJECTION_THRESHOLD:
        return 0.0

    trust = _error_to_trust(error)

    rating = feedback.get("rating")
    if rating is not None:
        try:
            rating = int(rating)
            rating_norm = _clamp((rating - 1) / 9)
            trust = 0.7 * trust + 0.3 * rating_norm
        except:
            pass

    return _clamp(trust)


# ─────────────────────────────────────────────
# DUPLICATE DETECTION (UPGRADED)
# ─────────────────────────────────────────────
def _duplicate_penalty(history: List[Dict]) -> float:

    signatures = [
        (round(_safe_float(fb.get("predicted")), 2),
         round(_safe_float(fb.get("actual")), 2))
        for fb in history
    ]

    counts = Counter(signatures)

    max_dup = max(counts.values()) if counts else 1

    if max_dup <= 2:
        return 1.0

    return max(0.5, 1.0 - (max_dup / len(history)))


# ─────────────────────────────────────────────
# TEMPORAL CONSISTENCY
# ─────────────────────────────────────────────
def _temporal_consistency(errors: List[float]) -> float:
    if len(errors) < 3:
        return 1.0

    diffs = [abs(errors[i] - errors[i - 1]) for i in range(1, len(errors))]
    avg_diff = sum(diffs) / len(diffs)

    return math.exp(-5 * avg_diff)


# ─────────────────────────────────────────────
# DRIFT DETECTION (NEW)
# ─────────────────────────────────────────────
def _detect_drift(errors: List[float]) -> float:
    if len(errors) < 5:
        return 1.0

    mid = len(errors) // 2
    first = sum(errors[:mid]) / max(mid, 1)
    second = sum(errors[mid:]) / max(len(errors) - mid, 1)

    drift = abs(first - second)

    return math.exp(-4 * drift)


# ─────────────────────────────────────────────
# ADVERSARIAL DETECTION (UPGRADED)
# ─────────────────────────────────────────────
def _detect_adversarial(errors: List[float]) -> float:

    if len(errors) < MIN_HISTORY:
        return 1.0

    high_error_ratio = sum(1 for e in errors if e > 0.4) / len(errors)

    oscillations = sum(
        1 for i in range(2, len(errors))
        if (errors[i] - errors[i-1]) * (errors[i-1] - errors[i-2]) < 0
    )

    osc_ratio = oscillations / len(errors)

    if high_error_ratio > 0.7:
        return 0.2

    if osc_ratio > 0.5:
        return 0.5

    return 1.0


# ─────────────────────────────────────────────
# AGGREGATED TRUST
# ─────────────────────────────────────────────
def compute_trust_score(
    feedback_history: Optional[List[Dict]],
    current_prediction: float
) -> float:

    if not feedback_history:
        return 0.5

    try:
        history = feedback_history[-MAX_HISTORY:]
        n = len(history)

        scores = []
        weights = []
        errors = []

        for i, fb in enumerate(history):

            pred = _safe_float(fb.get("predicted"))
            actual = _safe_float(fb.get("actual"))

            if not (_valid_range(pred) and _valid_range(actual)):
                continue

            err = abs(pred - actual)
            errors.append(err)

            trust = compute_trust_single(fb)

            weight = DECAY_FACTOR ** (n - i - 1)

            scores.append(trust * weight)
            weights.append(weight)

        if not weights:
            return 0.5

        avg_trust = sum(scores) / (sum(weights) + EPS)

        # ── stability
        mean_err = sum(errors) / len(errors)
        variance = sum((e - mean_err) ** 2 for e in errors) / len(errors)
        stability = math.exp(-variance * 6)

        # ── consistency
        consistency = _temporal_consistency(errors[-CONSISTENCY_WINDOW:])

        # ── drift
        drift = _detect_drift(errors)

        # ── adversarial
        adversarial = _detect_adversarial(errors)

        # ── duplicate penalty
        dup_penalty = _duplicate_penalty(history)

        # ── prediction alignment (FIXED)
        current_prediction = _safe_float(current_prediction)
        alignment = math.exp(-abs(current_prediction - mean_err) * 2)

        final = (
            0.4 * avg_trust +
            0.15 * stability +
            0.15 * consistency +
            0.1 * drift +
            0.1 * adversarial +
            0.1 * alignment
        ) * dup_penalty

        return _clamp(final)

    except Exception:
        return 0.5


# ─────────────────────────────────────────────
# FILTER VALID FEEDBACK
# ─────────────────────────────────────────────
def filter_valid_feedback(feedback_list: List[Dict]) -> List[Dict]:

    valid = []

    for fb in feedback_list:
        trust = compute_trust_single(fb)

        if trust > 0.25:
            valid.append(fb)

    return valid