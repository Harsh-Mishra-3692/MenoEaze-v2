# db_client.py — FINAL ELITE v2 (RESILIENT + IDEMPOTENT + SAFE)

import logging
import time
import hashlib
import threading
from typing import Dict, List, Any, Optional, Callable

from supabase import create_client, Client
from ml_engine.config import CONFIG

logger = logging.getLogger("menoeaze.db")

SUPABASE_URL = CONFIG["db"].url
SUPABASE_KEY = CONFIG["db"].key

_client: Optional[Client] = None
_client_lock = threading.Lock()

MAX_RETRIES = 3
BACKOFF_BASE = 0.3
TIMEOUT_WARN_MS = 500

FEATURES = 11
MAX_PAYLOAD_SIZE = 1000  # safety

# circuit breaker
_FAILURE_COUNT = 0
_FAILURE_LIMIT = 5
_CIRCUIT_OPEN = False
_CIRCUIT_RESET_TIME = 10
_LAST_FAILURE_TIME = 0


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
        x = float(v)
        return max(a, min(b, x))
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
    return isinstance(user_id, str) and 5 < len(user_id) < 128


def _hash_payload(payload: Dict):
    return hashlib.md5(str(payload).encode()).hexdigest()


def _validate_identifier(name: str):
    if not isinstance(name, str):
        return None
    name = name.strip()
    if not name.replace("_", "").isalnum():
        return None
    return name


# ─────────────────────────────────────────────
# CIRCUIT BREAKER
# ─────────────────────────────────────────────
def _check_circuit():
    global _CIRCUIT_OPEN, _LAST_FAILURE_TIME

    if not _CIRCUIT_OPEN:
        return True

    if time.time() - _LAST_FAILURE_TIME > _CIRCUIT_RESET_TIME:
        _CIRCUIT_OPEN = False
        return True

    return False


def _record_failure():
    global _FAILURE_COUNT, _CIRCUIT_OPEN, _LAST_FAILURE_TIME

    _FAILURE_COUNT += 1
    _LAST_FAILURE_TIME = time.time()

    if _FAILURE_COUNT >= _FAILURE_LIMIT:
        _CIRCUIT_OPEN = True
        logger.error("[DB] Circuit breaker OPEN")


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
            logger.error("[DB] Missing config")
            return None

        try:
            _client = create_client(SUPABASE_URL, SUPABASE_KEY)
            return _client
        except Exception as e:
            logger.error(f"[DB] init failed: {e}")
            return None


# ─────────────────────────────────────────────
# EXECUTION (WITH BACKOFF + CIRCUIT)
# ─────────────────────────────────────────────
def _execute(fn: Callable, op: str):

    if not _check_circuit():
        logger.error(f"[DB][{op}] circuit open, skipping")
        return None

    for attempt in range(MAX_RETRIES + 1):

        start = time.time()

        try:
            res = fn()

            latency = (time.time() - start) * 1000
            if latency > TIMEOUT_WARN_MS:
                logger.warning(f"[DB][{op}] slow: {latency:.1f}ms")

            return res

        except Exception as e:
            _record_failure()

            logger.error(f"[DB][{op}] attempt {attempt}: {e}")

            if attempt >= MAX_RETRIES:
                return None

            time.sleep(BACKOFF_BASE * (2 ** attempt))


# ─────────────────────────────────────────────
# BULK INSERT (CHUNK SAFE)
# ─────────────────────────────────────────────
def insert_bulk(table: str, payload: List[Dict]) -> bool:

    table = _validate_identifier(table)
    if not table or not payload:
        return False

    if len(payload) > MAX_PAYLOAD_SIZE:
        chunks = [
            payload[i:i + MAX_PAYLOAD_SIZE]
            for i in range(0, len(payload), MAX_PAYLOAD_SIZE)
        ]
    else:
        chunks = [payload]

    client = get_client()
    if not client:
        return False

    success = True

    for chunk in chunks:
        try:
            res = _execute(
                lambda: client.table(table).insert(chunk).execute(),
                f"insert_{table}"
            )

            if not res or not getattr(res, "data", None):
                success = False

        except Exception as e:
            logger.error(f"[DB] chunk insert failed: {e}")
            success = False

    return success


# ─────────────────────────────────────────────
# FETCH TABLE
# ─────────────────────────────────────────────
def fetch_table(table: str, filters: Dict = None, limit: int = 100) -> List[Dict]:

    table = _validate_identifier(table)
    if not table:
        return []

    client = get_client()
    if not client:
        return []

    try:
        query = client.table(table).select("*")

        if isinstance(filters, dict):
            for k, v in filters.items():
                query = query.eq(k, v)

        res = _execute(lambda: query.limit(limit).execute(), f"fetch_{table}")

        if not res or not hasattr(res, "data"):
            return []

        return res.data or []

    except Exception:
        return []


# ─────────────────────────────────────────────
# RPC
# ─────────────────────────────────────────────
def call_rpc(func: str, params: Dict = None) -> List[Dict]:

    func = _validate_identifier(func)
    if not func:
        return []

    client = get_client()
    if not client:
        return []

    try:
        res = _execute(
            lambda: client.rpc(func, params or {}).execute(),
            f"rpc_{func}"
        )

        if not res or not hasattr(res, "data"):
            return []

        return res.data or []

    except Exception:
        return []


# ─────────────────────────────────────────────
# DOMAIN SPECIFIC METHODS
# ─────────────────────────────────────────────
def insert_symptom_log(user_id: str, feature_vector: List[float], raw_text: str, emoji: str) -> bool:
    if not _valid_user(user_id): return False
    vec = _safe_vec(feature_vector)
    if not vec: return False
    payload = {
        "user_id": user_id,
        "feature_vector": vec,
        "raw_text": _safe_str(raw_text),
        "emoji": _safe_str(emoji, 10)
    }
    client = get_client()
    if not client: return False
    res = _execute(lambda: client.table("symptom_logs").insert(payload).execute(), "insert_symptom_log")
    return bool(res and getattr(res, "data", None))

def fetch_recent_logs(user_id: str) -> Optional[List[Dict]]:
    if not _valid_user(user_id): return []
    client = get_client()
    if not client: return None
    res = _execute(
        lambda: client.table("symptom_logs").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(10).execute(),
        "fetch_recent_logs"
    )
    if res is None:
        return None
    if hasattr(res, "data"):
        return list(reversed(res.data or []))
    return []

def insert_prediction(user_id: str, severity: float, confidence: float, reasoning: str) -> Optional[str]:
    if not _valid_user(user_id): return None
    payload = {
        "user_id": user_id,
        "severity": _safe_float(severity),
        "confidence": _safe_float(confidence),
        "reasoning": _safe_str(reasoning, 1000)
    }
    client = get_client()
    if not client: return None
    res = _execute(lambda: client.table("predictions").insert(payload).execute(), "insert_prediction")
    if res and getattr(res, "data", None) and len(res.data) > 0:
        return res.data[0].get("id")
    return None

def fetch_prediction_history(user_id: str) -> List[Dict]:
    return fetch_table("predictions", {"user_id": user_id}, limit=50)

def insert_feedback(user_id: str, prediction_id: str, predicted: float, actual: float, rating: int, trust_score: float) -> bool:
    if not _valid_user(user_id): return False
    payload = {
        "user_id": user_id,
        "prediction_id": _safe_str(prediction_id, 50),
        "predicted": _safe_float(predicted),
        "actual": _safe_float(actual),
        "rating": max(1, min(10, int(rating))),
        "trust_score": _safe_float(trust_score)
    }
    client = get_client()
    if not client: return False
    res = _execute(lambda: client.table("feedback").insert(payload).execute(), "insert_feedback")
    return bool(res and getattr(res, "data", None))

def fetch_feedback_history(user_id: str) -> List[Dict]:
    return fetch_table("feedback", {"user_id": user_id}, limit=50)

def insert_guardrail_log(user_id: str, **kwargs) -> bool:
    if not _valid_user(user_id): return False
    payload = {"user_id": user_id}
    for k, v in kwargs.items():
        payload[_safe_str(k, 50)] = _safe_str(v, 500)
    client = get_client()
    if not client: return False
    res = _execute(lambda: client.table("guardrail_logs").insert(payload).execute(), "insert_guardrail_log")
    return bool(res and getattr(res, "data", None))

def get_user_weights(user_id: str) -> Optional[Dict]:
    if not _valid_user(user_id): return None
    client = get_client()
    if not client: return None
    res = _execute(
        lambda: client.table("user_weights").select("*").eq("user_id", user_id).limit(1).execute(),
        "get_user_weights"
    )
    if res and hasattr(res, "data") and res.data:
        return res.data[0].get("weights")
    return None