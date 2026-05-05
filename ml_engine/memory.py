import time
import logging
from typing import Dict, List, Optional
from ml_engine.db_client import fetch_table, get_client

logger = logging.getLogger("menoeaze.memory")

RECENT_K = 8
OUTLIER_THRESHOLD = 0.6

def _safe_float(v, a=0.0, b=1.0):
    try:
        x = float(v)
        if x != x:
            return a
        return max(a, min(b, x))
    except:
        return a

def extract_user_signal(user_id: str) -> Dict[str, float | str]:
    if not user_id or len(user_id) < 3:
        return {"avg_severity": 0.0, "trend": "unknown", "volatility": 0.0, "signal_strength": 0.0}

    try:
        preds = fetch_table("predictions", {"user_id": user_id}, limit=RECENT_K)
        if not preds:
            return {"avg_severity": 0.0, "trend": "unknown", "volatility": 0.0, "signal_strength": 0.0}

        preds.reverse()

        severities = [_safe_float(p.get("severity", 0.0)) for p in preds]

        if len(severities) < 2:
            return {"avg_severity": severities[0] if severities else 0.0, "trend": "unknown", "volatility": 0.0, "signal_strength": 0.0}

        median = sorted(severities)[len(severities)//2]
        filtered = [s for s in severities if abs(s - median) < OUTLIER_THRESHOLD]
        if not filtered:
            filtered = severities

        weights = [0.9 ** i for i in reversed(range(len(filtered)))]
        avg = sum(s*w for s, w in zip(filtered, weights)) / max(sum(weights), 1)

        trend_val = filtered[-1] - filtered[0]
        if abs(trend_val) < 0.03:
            trend = "stable"
        elif trend_val > 0:
            trend = "worsening"
        else:
            trend = "improving"

        volatility = max(filtered) - min(filtered)
        strength = min(1.0, len(filtered) / RECENT_K) * (1 - volatility)

        return {
            "avg_severity": round(avg, 3),
            "trend": trend,
            "volatility": round(volatility, 3),
            "signal_strength": round(strength, 3)
        }
    except Exception as e:
        logger.error(f"[MEMORY] extraction failed: {e}")
        return {"avg_severity": 0.0, "trend": "unknown", "volatility": 0.0, "signal_strength": 0.0}

def get_full_history(user_id: str) -> List[Dict]:
    if not user_id or len(user_id) < 3:
        return []
    try:
        logs = fetch_table("symptom_logs", {"user_id": user_id}, limit=20)
        logs.reverse()
        
        history = []
        for l in logs:
            fv = l.get("feature_vector", [])
            sev = fv[0] if fv else 0.0
            history.append({
                "query": l.get("notes", ""),
                "severity": _safe_float(sev/10.0),
                "ts": l.get("created_at")
            })
        return history
    except:
        return []

def build_context_snippet(user_id: str) -> str:
    history = get_full_history(user_id)[-5:]
    if not history:
        return ""

    try:
        parts = []
        for h in history:
            q = str(h.get("query", ""))[:50]
            sev = _safe_float(h.get("severity", 0.0))
            if sev > 0:
                parts.append(f"{q} (sev={round(sev,2)})")
            elif q:
                parts.append(q)

        signal = extract_user_signal(user_id)
        summary = f"[avg={signal['avg_severity']} trend={signal['trend']} vol={signal['volatility']}]"
        
        return (" | ".join([p for p in parts if p] + [summary]))[:500]
    except:
        return ""

def add_interaction(*args, **kwargs):
    return True

def add_prediction_context(user_id: str, query: str, severity: float, confidence: float):
    """Persist prediction context to user_memory for long-term personalization."""
    try:
        from ml_engine.db_client import append_user_memory
        signal = extract_user_signal(user_id)
        append_user_memory(user_id, {
            "severity": severity,
            "trend": signal.get("trend", "unknown"),
            "meta": {"last_query": query[:100], "confidence": confidence}
        })
        return True
    except Exception as e:
        logger.error(f"[MEMORY] persist failed: {e}")
        return False