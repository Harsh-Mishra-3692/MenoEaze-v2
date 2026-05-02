# memory.py — FINAL ELITE (TEMPORAL + SIGNAL-AWARE + SAFE)

import time
import logging
import hashlib
from typing import Dict, List, Optional

from ml_engine.db_client import get_user_memory, append_user_memory

logger = logging.getLogger("menoeaze.memory")

MAX_HISTORY = 50
RECENT_K = 8
DEDUP_WINDOW = 6
MIN_USER_ID_LEN = 10

MAX_TEXT_LEN = 300
MIN_QUERY_LEN = 2

OUTLIER_THRESHOLD = 0.6


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────
def _valid_user(user_id: str) -> bool:
    return isinstance(user_id, str) and len(user_id) >= MIN_USER_ID_LEN


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _safe_float(v, a=0.0, b=1.0):
    try:
        x = float(v)
        if x != x:
            return a
        return max(a, min(b, x))
    except:
        return a


def _safe_str(v):
    try:
        return str(v).strip()[:MAX_TEXT_LEN]
    except:
        return ""


def _hash(text: str) -> str:
    return hashlib.md5(text.lower().strip().encode()).hexdigest()


# ─────────────────────────────────────────────
# SANITIZATION
# ─────────────────────────────────────────────
def _sanitize(entry: Dict) -> Dict:
    q = _safe_str(entry.get("query", ""))

    return {
        "ts": float(entry.get("ts", time.time())),
        "query": q,
        "severity": _safe_float(entry.get("severity", 0.0)),
        "confidence": _safe_float(entry.get("confidence", 0.0)),
        "meta": entry.get("meta", {}) if isinstance(entry.get("meta"), dict) else {},
        "hash": _hash(q)
    }


def _sanitize_history(history: List[Dict]) -> List[Dict]:
    out = []

    for h in history:
        if not isinstance(h, dict):
            continue

        e = _sanitize(h)
        if e["query"]:
            out.append(e)

    return out


# ─────────────────────────────────────────────
# ADVANCED DEDUP
# ─────────────────────────────────────────────
def _similar(a: str, b: str):
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _is_duplicate(history: List[Dict], query: str):
    h = _hash(query)

    for item in history[-DEDUP_WINDOW:]:
        if item["hash"] == h:
            return True

        if _similar(item["query"], query) > 0.85:
            return True

    return False


# ─────────────────────────────────────────────
# ADD INTERACTION
# ─────────────────────────────────────────────
def add_interaction(user_id: str, query: str, metadata: Optional[dict] = None):

    if not _valid_user(user_id):
        return False

    query = _safe_str(query)

    if len(query) < MIN_QUERY_LEN:
        return False

    try:
        history = _sanitize_history(get_user_memory(user_id))

        if _is_duplicate(history, query):
            return True

        entry = _sanitize({
            "ts": time.time(),
            "query": query,
            "meta": metadata or {}
        })

        return append_user_memory(user_id, entry)

    except Exception as e:
        logger.error(f"[Memory] add_interaction failed: {e}")
        return False


# ─────────────────────────────────────────────
# ADD PREDICTION
# ─────────────────────────────────────────────
def add_prediction_context(user_id, query, severity, confidence):

    if not _valid_user(user_id):
        return False

    try:
        entry = _sanitize({
            "ts": time.time(),
            "query": query,
            "severity": severity,
            "confidence": confidence
        })

        return append_user_memory(user_id, entry)

    except Exception:
        return False


# ─────────────────────────────────────────────
# FETCH
# ─────────────────────────────────────────────
def get_recent_history(user_id, k=RECENT_K):

    if not _valid_user(user_id):
        return []

    try:
        h = _sanitize_history(get_user_memory(user_id))
        h.sort(key=lambda x: x["ts"])
        return h[-k:]
    except:
        return []


def get_full_history(user_id):

    if not _valid_user(user_id):
        return []

    try:
        h = _sanitize_history(get_user_memory(user_id))
        h.sort(key=lambda x: x["ts"])
        return h[-MAX_HISTORY:]
    except:
        return []


# ─────────────────────────────────────────────
# SIGNAL EXTRACTION (UPGRADED)
# ─────────────────────────────────────────────
def extract_user_signal(user_id):

    history = get_recent_history(user_id, k=RECENT_K)

    if not history:
        return {
            "avg_severity": 0.0,
            "trend": "unknown",
            "volatility": 0.0,
            "signal_strength": 0.0
        }

    try:
        severities = [_safe_float(h["severity"]) for h in history]

        if len(severities) < 2:
            return {
                "avg_severity": 0.0,
                "trend": "unknown",
                "volatility": 0.0,
                "signal_strength": 0.0
            }

        # ── outlier removal
        median = sorted(severities)[len(severities)//2]
        filtered = [s for s in severities if abs(s - median) < OUTLIER_THRESHOLD]
        if not filtered:
            filtered = severities

        # ── temporal weighting
        weights = [0.9 ** i for i in reversed(range(len(filtered)))]
        avg = sum(s*w for s, w in zip(filtered, weights)) / max(sum(weights), 1)

        trend_val = filtered[-1] - filtered[0]

        if abs(trend_val) < 0.03:
            trend = "stable"
        elif trend_val > 0:
            trend = "increasing"
        else:
            trend = "decreasing"

        volatility = max(filtered) - min(filtered)

        # ── signal strength
        strength = min(1.0, len(filtered) / RECENT_K) * (1 - volatility)

        return {
            "avg_severity": round(avg, 3),
            "trend": trend,
            "volatility": round(volatility, 3),
            "signal_strength": round(strength, 3)
        }

    except Exception:
        return {
            "avg_severity": 0.0,
            "trend": "unknown",
            "volatility": 0.0,
            "signal_strength": 0.0
        }


# ─────────────────────────────────────────────
# CONTEXT BUILDER (STRUCTURED)
# ─────────────────────────────────────────────
def build_context_snippet(user_id):

    history = get_recent_history(user_id, k=5)

    if not history:
        return ""

    try:
        parts = []

        for h in history:
            q = _safe_str(h["query"])
            sev = _safe_float(h["severity"])

            if sev > 0:
                parts.append(f"{q} (sev={round(sev,2)})")
            else:
                parts.append(q)

        signal = extract_user_signal(user_id)

        summary = (
            f"[avg={signal['avg_severity']} "
            f"trend={signal['trend']} "
            f"vol={signal['volatility']}]"
        )

        return (" | ".join(parts + [summary]))[:500]

    except:
        return ""