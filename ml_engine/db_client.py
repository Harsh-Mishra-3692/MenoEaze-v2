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