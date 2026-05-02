# doctor_recommender.py — FINAL HARDENED (CLINICAL + SAFE + EXPLAINABLE)

from typing import Dict, List, Optional
import math

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
HIGH_SEVERITY = 0.85
MODERATE_SEVERITY = 0.7
LOW_SEVERITY = 0.5

PERSISTENT_THRESHOLD = 0.6
TREND_STRONG_THRESHOLD = 0.1

MAX_HISTORY = 10

EMERGENCY_KEYWORDS = ["chest pain", "fainting", "loss of consciousness"]

# ─────────────────────────────────────────────
# SAFE HELPERS
# ─────────────────────────────────────────────
def _safe_float(x, default=0.5):
    try:
        x = float(x)
        if not (x == x):  # NaN
            return default
        return x
    except:
        return default


def _clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def _normalize_trend(trend: Optional[str]) -> str:
    if not isinstance(trend, str):
        return "unknown"
    t = trend.lower()
    return t if t in ["increasing", "decreasing", "stable"] else "unknown"


# ─────────────────────────────────────────────
# HISTORY ANALYSIS (ROBUST)
# ─────────────────────────────────────────────
def _analyze_history(history: List[Dict]) -> Dict:

    if not history:
        return {
            "avg": 0.5,
            "max": 0.5,
            "trend_strength": 0.0,
            "persistent": False,
            "volatility": 0.0
        }

    try:
        recent = history[-MAX_HISTORY:]
        values = [_safe_float(h.get("severity", 0.5)) for h in recent]

        # remove extreme outliers
        median = sorted(values)[len(values)//2]
        values = [v for v in values if abs(v - median) < 0.5] or values

        avg = sum(values) / len(values)
        mx = max(values)

        trend_strength = values[-1] - values[0] if len(values) > 2 else 0.0

        var = sum((v - avg) ** 2 for v in values) / len(values)
        volatility = math.sqrt(var)

        return {
            "avg": avg,
            "max": mx,
            "trend_strength": trend_strength,
            "persistent": avg > PERSISTENT_THRESHOLD,
            "volatility": volatility
        }

    except Exception:
        return {
            "avg": 0.5,
            "max": 0.5,
            "trend_strength": 0.0,
            "persistent": False,
            "volatility": 0.0
        }


# ─────────────────────────────────────────────
# EMERGENCY DETECTION (NEW)
# ─────────────────────────────────────────────
def _detect_emergency(query: str, severity: float) -> bool:

    q = (query or "").lower()

    if severity >= 0.95:
        return True

    for kw in EMERGENCY_KEYWORDS:
        if kw in q:
            return True

    return False


# ─────────────────────────────────────────────
# SPECIALIST SELECTION (SAFER)
# ─────────────────────────────────────────────
def _select_specialist(severity: float, query: Optional[str]) -> str:

    q = (query or "").lower()

    if "heart" in q or "chest" in q:
        return "Cardiology"

    if "anxiety" in q or "panic" in q:
        return "Psychiatry"

    if severity >= MODERATE_SEVERITY:
        return "Gynecologist / Endocrinologist"

    return "General Physician"


# ─────────────────────────────────────────────
# RISK SCORING (NONLINEAR)
# ─────────────────────────────────────────────
def _compute_risk_score(severity, trend, hist, confidence):

    # nonlinear severity emphasis
    sev_score = severity ** 1.5

    trend_score = 0.0
    if trend == "increasing":
        trend_score = 0.2
    elif trend == "decreasing":
        trend_score = -0.1

    persistence_score = 0.15 if hist["persistent"] else 0.0

    volatility_penalty = min(0.1, hist["volatility"])

    uncertainty_bonus = (1 - confidence) * 0.15

    score = (
        0.5 * sev_score +
        trend_score +
        persistence_score +
        volatility_penalty +
        uncertainty_bonus
    )

    return _clamp(score)


# ─────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────
def recommend_doctor(
    severity: float,
    trend: Optional[str],
    history: Optional[List[Dict]],
    query: Optional[str] = None,
    confidence: float = 0.5
) -> Dict:

    try:
        severity = _clamp(_safe_float(severity))
        confidence = _clamp(_safe_float(confidence))
        trend = _normalize_trend(trend)
        history = history or []

        # ───────── EMERGENCY OVERRIDE
        if _detect_emergency(query, severity):
            return {
                "recommend": True,
                "urgency": "immediate",
                "specialist": "Emergency Care",
                "reason": "Potential emergency symptoms detected",
                "score": 1.0,
                "signals": {}
            }

        hist = _analyze_history(history)
        score = _compute_risk_score(severity, trend, hist, confidence)

        # ───────── DECISION
        if score >= 0.8:
            urgency = "immediate"
            recommend = True
            reason = "High severity and/or worsening trend"

        elif score >= 0.65:
            urgency = "soon"
            recommend = True
            reason = "Persistent or increasing symptoms"

        elif score >= 0.5:
            urgency = "moderate"
            recommend = True
            reason = "Moderate symptoms with clinical relevance"

        elif score >= 0.35:
            urgency = "optional"
            recommend = False
            reason = "Mild symptoms"

        else:
            urgency = "none"
            recommend = False
            reason = ""

        specialist = _select_specialist(severity, query) if recommend else None

        return {
            "recommend": recommend,
            "urgency": urgency,
            "specialist": specialist,
            "reason": reason,
            "score": round(score, 3),
            "signals": {
                "avg_severity": round(hist["avg"], 3),
                "trend_strength": round(hist["trend_strength"], 3),
                "volatility": round(hist["volatility"], 3),
                "confidence": confidence
            }
        }

    except Exception:
        return {
            "recommend": True,
            "urgency": "moderate",
            "specialist": "General Physician",
            "reason": "Unable to assess safely — consult a doctor",
            "score": 0.5,
            "signals": {}
        }