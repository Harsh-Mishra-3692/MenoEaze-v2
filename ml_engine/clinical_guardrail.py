# clinical_guardrail.py — PRODUCTION CLINICAL SAFETY ENGINE (MENOEAZE)

import re
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("menoeaze.guardrail")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
HIGH_RISK_THRESHOLD = 0.85
MEDIUM_RISK_THRESHOLD = 0.65

TREND_WINDOW = 4
TREND_ESCALATION_DELTA = 0.15
SUSTAINED_HIGH_COUNT = 3

# ─────────────────────────────────────────────
# PRECOMPILED PATTERNS (PERF FIX)
# ─────────────────────────────────────────────
RED_FLAG_PATTERNS = [
    re.compile(p) for p in [
        r"\bchest pain\b",
        r"\bshortness of breath\b",
        r"\bfaint(ing)?\b",
        r"\bblackout\b",
        r"\bloss of consciousness\b",
        r"\bstroke\b",
        r"\bheart attack\b"
    ]
]

URGENT_PATTERNS = [
    re.compile(p) for p in [
        r"\bpalpitations?\b",
        r"\birregular heartbeat\b",
        r"\bsevere anxiety\b",
        r"\bdizziness\b"
    ]
]

DOMAIN_PATTERNS = [
    re.compile(p) for p in [
        r"\bmenopause\b",
        r"\bhot flashes?\b",
        r"\bnight sweats?\b",
        r"\bhormone\b",
        r"\bperiod\b",
        r"\bperimenopause\b"
    ]
]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _normalize(text: str) -> str:
    return (text or "").lower().strip()


def _match(text: str, patterns: List[re.Pattern]) -> bool:
    return any(p.search(text) for p in patterns)


def _risk_level(severity: float) -> str:
    if severity >= HIGH_RISK_THRESHOLD:
        return "high"
    elif severity >= MEDIUM_RISK_THRESHOLD:
        return "medium"
    return "low"


# ─────────────────────────────────────────────
# SAFE HISTORY EXTRACTION
# ─────────────────────────────────────────────
def _extract_history(user_history: Optional[List[Dict]]) -> List[float]:
    if not user_history:
        return []

    vals = []

    for h in user_history[-10:]:
        try:
            sev = float(h.get("severity", 0))
            vals.append(max(0.0, min(1.0, sev)))
        except:
            continue

    return vals


# ─────────────────────────────────────────────
# TREND
# ─────────────────────────────────────────────
def _trend(history: List[float]) -> Dict[str, Any]:
    if len(history) < 2:
        return {"trend": "stable", "escalate": False}

    recent = history[-TREND_WINDOW:]
    delta = recent[-1] - recent[0]

    if delta > TREND_ESCALATION_DELTA:
        return {"trend": "increasing", "escalate": True}

    if delta < -TREND_ESCALATION_DELTA:
        return {"trend": "decreasing", "escalate": False}

    return {"trend": "stable", "escalate": False}


def _sustained_high(history: List[float]) -> bool:
    if len(history) < SUSTAINED_HIGH_COUNT:
        return False

    return all(h >= MEDIUM_RISK_THRESHOLD for h in history[-SUSTAINED_HIGH_COUNT:])


# ─────────────────────────────────────────────
# DOCTOR LOGIC
# ─────────────────────────────────────────────
def _doctor(severity: float, trend_info: Dict, history: List[float]) -> Dict:

    if severity >= HIGH_RISK_THRESHOLD:
        return {
            "doctor_recommended": True,
            "urgency": "immediate",
            "message": "Seek immediate medical attention."
        }

    if trend_info["escalate"]:
        return {
            "doctor_recommended": True,
            "urgency": "soon",
            "message": "Symptoms worsening. Consult a doctor soon."
        }

    if _sustained_high(history):
        return {
            "doctor_recommended": True,
            "urgency": "moderate",
            "message": "Persistent symptoms detected. Medical consultation recommended."
        }

    if severity >= MEDIUM_RISK_THRESHOLD:
        return {
            "doctor_recommended": True,
            "urgency": "optional",
            "message": "Consider consulting a healthcare provider."
        }

    return {
        "doctor_recommended": False,
        "urgency": "none",
        "message": ""
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def apply_guardrail(
    query: str,
    severity: float,
    user_history: Optional[List[Dict]] = None
) -> Dict[str, Any]:

    try:
        q = _normalize(query)
        severity = max(0.0, min(1.0, float(severity)))

        history_vals = _extract_history(user_history)
        trend_info = _trend(history_vals)

        risk = _risk_level(severity)

        # ── DOMAIN CHECK (SOFT — frontend handles hard filtering) ──
        if not _match(q, DOMAIN_PATTERNS):
            return {
                "safe": True,
                "override": False,
                "risk_level": "low",
                "risk_score": severity,
                "trend": "unknown",
                "requires_attention": False,
                "flags": ["off_domain_soft"],
                "message": None
            }

        # ── RED FLAG (HIGHEST PRIORITY) ─────
        if _match(q, RED_FLAG_PATTERNS):
            return {
                "override": True,
                "risk_level": "high",
                "risk_score": severity,
                "trend": "unknown",
                "requires_attention": True,
                "doctor_recommended": True,
                "urgency": "immediate",
                "message": "Emergency symptoms detected. Seek immediate medical help."
            }

        # ── TREND ESCALATION ────────────────
        if trend_info["escalate"] and risk != "high":
            risk = "high"

        # ── DOCTOR LOGIC ───────────────────
        doctor = _doctor(severity, trend_info, history_vals)

        response = {
            "override": risk == "high",
            "risk_level": risk,
            "risk_score": round(severity, 4),
            "trend": trend_info["trend"],
            "requires_attention": risk != "low",
            **doctor
        }

        return response

    except Exception as e:
        logger.error(f"[Guardrail] failure: {e}")

        return {
            "override": True,
            "risk_level": "high",
            "risk_score": 1.0,
            "trend": "unknown",
            "requires_attention": True,
            "doctor_recommended": True,
            "urgency": "immediate",
            "message": "Unable to assess safely. Please consult a doctor."
        }