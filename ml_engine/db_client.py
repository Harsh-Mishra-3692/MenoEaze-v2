# db_client.py — PRODUCTION v7 (L7/L9 HARDENED | RAILWAY SAFE)

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

# 🔴 NEW (critical)
CIRCUIT_FAIL_THRESHOLD = 5
CIRCUIT_RESET_TIME = 30

_cache = {}
_cache_lock = threading.Lock()
CACHE_TTL = 5  # seconds

_fail_count = 0
_last_fail_time = 0

FEATURES = 11


# ─────────────────────────────────────────────
# CIRCUIT BREAKER
# ─────────────────────────────────────────────
def _circuit_open():
    global _fail_count, _last_fail_time

    if _fail_count < CIRCUIT_FAIL_THRESHOLD:
        return False

    if time.time() - _last_fail_time > CIRCUIT_RESET_TIME:
        _fail_count = 0
        return False

    return True


def _record_failure():
    global _fail_count, _last_fail_time
    _fail_count += 1
    _last_fail_time = time.time()


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
        return [float(x) if x == x else 0.0 for x in v]
    except:
        return None


def _valid_user(user_id: str):
    return isinstance(user_id, str) and len(user_id) >= 3


# ─────────────────────────────────────────────
# CLIENT (ONCE ONLY)
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
        except Exception as e:
            logger.error(f"[DB] init failed: {e}")
            return None

    return _client


# ─────────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────────
def _get_cache(key):
    with _cache_lock:
        item = _cache.get(key)
        if not item:
            return None
        if time.time() - item["time"] > CACHE_TTL:
            del _cache[key]
            return None
        return item["value"]


def _set_cache(key, value):
    with _cache_lock:
        _cache[key] = {"value": value, "time": time.time()}


# ─────────────────────────────────────────────
# EXECUTION WRAPPER
# ─────────────────────────────────────────────
def _execute(fn: Callable, op: str):

    if _circuit_open():
        logger.warning(f"[DB][{op}] Circuit open — skipping")
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
            logger.error(f"[DB][{op}] attempt {attempt}: {e}")
            _record_failure()

            if attempt >= MAX_RETRIES:
                return None

            time.sleep(BACKOFF_BASE * (2 ** attempt))


# ─────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────

def fetch_recent_logs(user_id: str, limit: int = 10):

    cache_key = f"logs:{user_id}:{limit}"
    cached = _get_cache(cache_key)
    if cached:
        return cached

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

    data = getattr(res, "data", []) if res else []
    _set_cache(cache_key, data)

    return data


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


def fetch_table(table_name: str, filters: dict = None, limit: int = 50):

    cache_key = f"{table_name}:{str(filters)}:{limit}"
    cached = _get_cache(cache_key)
    if cached:
        return cached

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
        data = getattr(res, "data", []) or []

        _set_cache(cache_key, data)

        return data

    except Exception as e:
        logger.error(f"[DB][FETCH] {e}")
        return []


def call_rpc(function_name: str, params: dict):

    if _circuit_open():
        return None

    client = get_client()
    if not client:
        return None

    try:
        res = client.rpc(function_name, params).execute()
        return getattr(res, "data", None)
    except Exception as e:
        logger.error(f"[DB][RPC] {e}")
        _record_failure()
        return None


# ─────────────────────────────────────────────
# REMAINING FUNCTIONS (UNCHANGED CONTRACT)
# ─────────────────────────────────────────────

def get_user_weights(user_id: str) -> Dict:
    data = fetch_table("user_weights", {"user_id": user_id}, 1)
    if not data:
        return {}
    row = data[0]
    return {
        "baseline_offset": row.get("baseline_offset", 0.0),
        "volatility_multiplier": row.get("volatility_multiplier", 1.0),
    }


def fetch_feedback_history(user_id: str, limit: int = 20):
    return fetch_table("feedback", {"user_id": user_id}, limit)
