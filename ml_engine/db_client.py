# db_client.py — ELITE v6 (STABLE | CONTRACT-COMPLETE | DEMO-SAFE)

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Callable

from supabase import create_client, Client
from ml_engine.config import CONFIG

logger = logging.getLogger("menoeaze.db")

SUPABASE_URL = CONFIG["db"].url
SUPABASE_KEY = CONFIG["db"].key

_client: Optional[Client] = None
_client_lock = threading.Lock()

MAX_RETRIES = 2
BACKOFF_BASE = 0.25
TIMEOUT_WARN_MS = 400

FEATURES = 11

# ─────────────────────────────────────────────
# SAFE HELPERS
# ─────────────────────────────────────────────
def _safe_str(v, max_len=255):
    try:
        return str(v).strip()[:max_len]
    except:
        return ""


def _safe_float(v, a=0.0, b=1.0):
    try:
        return max(a, min(b, float(v)))
    except:
        return a


def _safe_vec(v):
    if not isinstance(v, list) or len(v) != FEATURES:
        return None
    try:
        return [float(x) for x in v]
    except:
        return None


def _valid_user(user_id: str):
    return isinstance(user_id, str) and len(user_id) >= 3


# ─────────────────────────────────────────────
# CLIENT
# ─────────────────────────────────────────────
def get_client() -> Optional[Client]:
    global _client

    if _client:
        return _client

    with _client_lock:
        if _client:
            return _client

        if not SUPABASE_URL or not SUPABASE_KEY:
            logger.error("[DB] Missing Supabase config")
            return None

        try:
            _client = create_client(SUPABASE_URL, SUPABASE_KEY)
            logger.info("[DB] Supabase client initialized")
            return _client
        except Exception as e:
            logger.error(f"[DB] init failed: {e}")
            return None


# ─────────────────────────────────────────────
# EXECUTION WRAPPER
# ─────────────────────────────────────────────
def _execute(fn: Callable, op: str):
    for attempt in range(MAX_RETRIES + 1):
        start = time.time()

        try:
            res = fn()

            latency = (time.time() - start) * 1000
            if latency > TIMEOUT_WARN_MS:
                logger.warning(f"[DB][{op}] slow: {latency:.1f}ms")

            return res

        except Exception as e:
            logger.error(f"[DB][{op}] attempt {attempt}: {e}")

            if attempt >= MAX_RETRIES:
                return None

            time.sleep(BACKOFF_BASE * (2 ** attempt))


# ─────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────

def insert_symptom_log(user_id: str, feature_vector: list, raw_text: str = "", emoji: str = ""):
    if not _valid_user(user_id):
        return False

    fv = _safe_vec(feature_vector)
    if fv is None:
        return False

    client = get_client()
    if not client:
        return False

    payload = {
        "user_id": user_id,
        "feature_vector": fv,
        "notes": _safe_str(raw_text, 300),
        "emoji": _safe_str(emoji, 10),
    }

    res = _execute(
        lambda: client.table("symptom_logs").insert(payload).execute(),
        "insert_symptom_log"
    )

    return bool(res and getattr(res, "data", None))


def fetch_recent_logs(user_id: str, limit: int = 10):
    client = get_client()
    if not client:
        return []

    res = _execute(
        lambda: client.table("symptom_logs")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute(),
        "fetch_recent_logs"
    )

    return getattr(res, "data", []) if res else []


def insert_prediction(user_id: str, severity: float, confidence: float, reasoning: str):
    client = get_client()
    if not client:
        return None

    payload = {
        "user_id": user_id,
        "severity": _safe_float(severity),
        "confidence": _safe_float(confidence),
    }

    res = _execute(
        lambda: client.table("predictions").insert(payload).execute(),
        "insert_prediction"
    )

    if res and getattr(res, "data", None):
        return res.data[0].get("id")

    return None


def insert_feedback(user_id: str, prediction_id: str, predicted: float, actual: float, rating: int, trust_score: float):
    client = get_client()
    if not client:
        return False

    payload = {
        "user_id": user_id,
        "prediction_id": prediction_id,
        "predicted": _safe_float(predicted),
        "actual": _safe_float(actual),
        "rating": int(max(1, min(10, rating))),
        "trust_score": _safe_float(trust_score),
    }

    res = _execute(
        lambda: client.table("feedback").insert(payload).execute(),
        "insert_feedback"
    )

    return bool(res and getattr(res, "data", None))


def insert_guardrail_log(user_id: str, query: str, severity: float, message: str):
    client = get_client()
    if not client:
        return False

    payload = {
        "user_id": user_id,
        "query": _safe_str(query, 300),
        "answer": _safe_str(message, 500),
        "sources": None,
    }

    _execute(
        lambda: client.table("rag_logs").insert(payload).execute(),
        "guardrail_log"
    )

    return True


# ─────────────────────────────────────────────
# GENERIC
# ─────────────────────────────────────────────

def fetch_table(table_name: str, filters: dict = None, limit: int = 50):
    client = get_client()
    if not client:
        return []

    try:
        query = client.table(table_name).select("*")

        if filters:
            for k, v in filters.items():
                query = query.eq(k, v)

        if limit:
            query = query.limit(limit)

        res = query.execute()
        return getattr(res, "data", []) or []

    except Exception as e:
        logger.error(f"[DB][FETCH] {e}")
        return []


# ─────────────────────────────────────────────
# REQUIRED INTERFACES (CRITICAL)
# ─────────────────────────────────────────────

def insert_bulk(table_name: str, batch: List[Dict]):
    client = get_client()
    if not client:
        return False

    res = _execute(
        lambda: client.table(table_name).insert(batch).execute(),
        f"insert_bulk_{table_name}"
    )

    return bool(res and getattr(res, "data", None))


def call_rpc(function_name: str, params: dict):
    client = get_client()
    if not client:
        return None

    try:
        res = client.rpc(function_name, params).execute()
        return getattr(res, "data", None)
    except Exception as e:
        logger.error(f"[DB][RPC] {e}")
        return None


def get_user_weights(user_id: str) -> Dict:
    data = fetch_table("user_weights", {"user_id": user_id}, 1)
    if not data:
        return {}
    row = data[0]
    # Reconstruct weights dict from DB columns for personalization adapter
    return {
        "baseline_offset": row.get("baseline_offset", 0.0),
        "volatility_multiplier": row.get("volatility_multiplier", 1.0),
    }


def fetch_feedback_history(user_id: str, limit: int = 20):
    return fetch_table("feedback", {"user_id": user_id}, limit)


def update_user_weights(user_id: str, new_weights: Dict[str, float], trust_score: float = 0.5) -> bool:
    client = get_client()
    if not client:
        return False

    payload = {
        "user_id": user_id,
        "baseline_offset": _safe_float(new_weights.get("bias", 0.0), -1.0, 1.0),
        "volatility_multiplier": _safe_float(new_weights.get("temporal_weight", 1.0), 0.0, 5.0),
    }

    res = _execute(
        lambda: client.table("user_weights").upsert(payload, on_conflict="user_id").execute(),
        "update_user_weights"
    )

    return bool(res)


def get_user_stats(user_id: str) -> Dict[str, Any]:
    client = get_client()
    if not client:
        return {}

    try:
        # Total Logs
        count_res = client.table("symptom_logs").select("id", count="exact").eq("user_id", user_id).execute()
        total_logs = count_res.count if hasattr(count_res, "count") else 0

        # Avg Severity from user_memory (pre-computed)
        memory = fetch_table("user_memory", {"user_id": user_id}, 1)
        avg_severity = memory[0].get("avg_severity", 0.0) if memory else 0.0

        # Last Log Date
        last_log = client.table("symptom_logs").select("created_at").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
        last_date = last_log.data[0].get("created_at") if last_log and last_log.data else None

        # Logs this week (simple count)
        recent = fetch_recent_logs(user_id, 20)
        
        # Chart Data (last 14 days/logs)
        chart_data = []
        for l in reversed(recent[:14]):
            fv = l.get("feature_vector", [0]*11)
            # severity is first feature, mood is second in our 11-feature schema (usually)
            # Actually, let's just use the feature vector indexes if we know them.
            # But we can also just return the logs and let frontend decide.
            chart_data.append({
                "date": l.get("created_at"),
                "severity": fv[0] if fv else 0,
                "mood": fv[1] if fv and len(fv) > 1 else 0
            })

        return {
            "total_logs": total_logs,
            "avg_severity": round(avg_severity, 2),
            "last_log_at": last_date,
            "recent_count": len(recent),
            "history": chart_data
        }
    except Exception as e:
        logger.error(f"[DB][STATS] {e}")
        return {}


def get_user_memory(user_id: str):
    data = fetch_table("user_memory", {"user_id": user_id}, 1)
    if not data:
        return []
    # Return as list to match existing consumer expectations
    return data


def append_user_memory(user_id: str, data: dict):
    client = get_client()
    if not client:
        return False

    # user_memory has UNIQUE(user_id) — must upsert with schema-compatible fields
    avg_sev = _safe_float(data.get("severity", 0.0))
    trend = str(data.get("trend", data.get("meta", {}).get("trend", "unknown")))[:20]

    payload = {
        "user_id": user_id,
        "avg_severity": avg_sev,
        "historical_trend": trend,
    }

    res = _execute(
        lambda: client.table("user_memory").upsert(payload, on_conflict="user_id").execute(),
        "append_memory"
    )

    return bool(res)