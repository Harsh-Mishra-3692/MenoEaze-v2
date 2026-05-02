# trust_filter.py — FINAL HARDENED (CLINICAL + ROBUST + ADVERSARIAL SAFE)

import math
from typing import List, Dict, Optional

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MAX_ERROR = 1.0
REJECTION_THRESHOLD = 0.6
MIN_HISTORY = 3
DECAY_FACTOR = 0.9

CONSISTENCY_WINDOW = 5
DUPLICATE_TOLERANCE = 0.01

MIN_VALID_VALUE = 0.0
MAX_VALID_VALUE = 1.0

# ─────────────────────────────────────────────
# SAFE HELPERS
# ─────────────────────────────────────────────
def _safe_float(x, default=0.5):
    try:
        x = float(x)
        if not (x == x):  # NaN check
            return default
        return x
    except:
        return default


def _clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def _valid_range(x):
    return MIN_VALID_VALUE <= x <= MAX_VALID_VALUE


# ─────────────────────────────────────────────
# NONLINEAR ERROR SCORING
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

    base = _error_to_trust(error)

    rating = feedback.get("rating")

    if rating is not None:
        try:
            rating = int(rating)
            rating_norm = (rating - 1) / 9
            rating_norm = _clamp(rating_norm)

            base = 0.7 * base + 0.3 * rating_norm
        except:
            pass

    return _clamp(base)


# ─────────────────────────────────────────────
# DUPLICATE DETECTION
# ─────────────────────────────────────────────
def _is_duplicate_sequence(errors: List[float]) -> bool:
    if len(errors) < 3:
        return False

    return max(errors) - min(errors) < DUPLICATE_TOLERANCE


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
# ADVERSARIAL DETECTION
# ─────────────────────────────────────────────
def _detect_adversarial(errors: List[float]) -> float:

    if len(errors) < MIN_HISTORY:
        return 1.0

    high_error_ratio = sum(1 for e in errors if e > 0.4) / len(errors)

    # oscillation detection (important)
    sign_changes = sum(
        1 for i in range(2, len(errors))
        if (errors[i] - errors[i-1]) * (errors[i-1] - errors[i-2]) < 0
    )

    oscillation_ratio = sign_changes / max(len(errors), 1)

    if high_error_ratio > 0.7:
        return 0.3

    if oscillation_ratio > 0.5:
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
        weights = []
        scores = []
        errors = []

        history = feedback_history[-20:]
        n = len(history)

        for i, fb in enumerate(history):

            pred = _safe_float(fb.get("predicted"))
            actual = _safe_float(fb.get("actual"))

            if not (_valid_range(pred) and _valid_range(actual)):
                continue

            err = abs(pred - actual)
            errors.append(err)

            trust = compute_trust_single(fb)

            # correct decay: recent = higher weight
            weight = DECAY_FACTOR ** (n - i - 1)

            scores.append(trust * weight)
            weights.append(weight)

        if not weights or sum(weights) == 0:
            return 0.5

        avg_trust = sum(scores) / sum(weights)

        # ── stability
        mean_err = sum(errors) / len(errors)
        variance = sum((e - mean_err) ** 2 for e in errors) / len(errors)
        stability = math.exp(-variance * 6)

        # ── consistency
        consistency = _temporal_consistency(errors[-CONSISTENCY_WINDOW:])

        # ── adversarial
        adversarial = _detect_adversarial(errors)

        # ── duplicate
        duplicate_penalty = 0.5 if _is_duplicate_sequence(errors) else 1.0

        # ── confidence-aware adjustment (NEW)
        current_prediction = _safe_float(current_prediction)
        confidence_factor = 1.0 - abs(current_prediction - mean_err)
        confidence_factor = _clamp(confidence_factor, 0.5, 1.0)

        final = (
            0.45 * avg_trust +
            0.2 * stability +
            0.15 * consistency +
            0.1 * adversarial +
            0.1 * confidence_factor
        ) * duplicate_penalty

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