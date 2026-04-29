# db_client.py — ELITE (PRODUCTION + SAFE + CENTRALIZED CONFIG)

import time
import logging
import threading
from typing import Dict, List, Any, Optional, Callable

from supabase import create_client, Client

# ✅ CENTRAL CONFIG
from ml_engine.config import CONFIG

logger = logging.getLogger("menoeaze.db")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SUPABASE_URL = CONFIG["db"].url
SUPABASE_KEY = CONFIG["db"].key

MAX_RETRIES = 3
BASE_DELAY = 0.5
MAX_DELAY = 3.0
TIMEOUT = 5.0

CIRCUIT_BREAKER_THRESHOLD = 5
CIRCUIT_BREAKER_RESET = 30

# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────
_supabase: Optional[Client] = None
_lock = threading.Lock()

_failure_count = 0
_last_failure_time = 0


# ─────────────────────────────────────────────
# CLIENT INIT
# ─────────────────────────────────────────────
def _init_client() -> Optional[Client]:
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        client.table("feedback").select("id").limit(1).execute()
        logger.info("[DB] Connected to Supabase")
        return client
    except Exception as e:
        logger.error(f"[DB] Init failed: {e}")
        return None


def _get_client() -> Optional[Client]:
    global _supabase

    if _supabase:
        return _supabase

    with _lock:
        if not _supabase:
            _supabase = _init_client()

    return _supabase


# ─────────────────────────────────────────────
# CIRCUIT BREAKER (THREAD-SAFE)
# ─────────────────────────────────────────────
def _circuit_open() -> bool:
    global _failure_count, _last_failure_time

    with _lock:
        if _failure_count < CIRCUIT_BREAKER_THRESHOLD:
            return False

        if time.time() - _last_failure_time > CIRCUIT_BREAKER_RESET:
            _failure_count = 0
            return False

        return True


def _record_failure():
    global _failure_count, _last_failure_time
    with _lock:
        _failure_count += 1
        _last_failure_time = time.time()


def _record_success():
    global _failure_count
    with _lock:
        _failure_count = 0


# ─────────────────────────────────────────────
# RETRY + TIMEOUT
# ─────────────────────────────────────────────
def _retry(operation: Callable):

    if _circuit_open():
        logger.error("[DB] Circuit breaker OPEN — skipping")
        return None

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            start = time.time()
            result = operation()

            if time.time() - start > TIMEOUT:
                raise TimeoutError("DB operation timeout")

            _record_success()
            return result

        except Exception as e:
            last_error = e
            _record_failure()

            delay = min(BASE_DELAY * (2 ** (attempt - 1)), MAX_DELAY)
            logger.warning(f"[DB] Attempt {attempt} failed → {e}")

            time.sleep(delay)

    logger.error(f"[DB] All retries failed: {last_error}")
    return None


# ─────────────────────────────────────────────
# SAFE EXECUTION
# ─────────────────────────────────────────────
def _execute(op: Callable):

    client = _get_client()

    if not client:
        logger.error("[DB] No client available")
        return None

    def safe_op():
        return op(client)

    start = time.time()
    res = _retry(safe_op)
    latency = time.time() - start

    logger.info(f"[DB] latency={latency:.3f}s")

    if not res:
        return None

    return getattr(res, "data", res)


# ─────────────────────────────────────────────
# INSERT FEEDBACK
# ─────────────────────────────────────────────
def insert_feedback(record: Dict[str, Any]) -> bool:

    if not isinstance(record, dict):
        logger.error("[DB] Invalid record format")
        return False

    required = ["user_id", "predicted", "actual", "error"]
    for k in required:
        if k not in record:
            logger.error(f"[DB] Missing field: {k}")
            return False

    def op(client):
        return client.table("feedback").insert(record).execute()

    return _execute(op) is not None


# ─────────────────────────────────────────────
# BULK INSERT
# ─────────────────────────────────────────────
def insert_bulk(table: str, rows: List[Dict[str, Any]]) -> bool:

    if not rows:
        return True

    def op(client):
        return client.table(table).insert(rows).execute()

    return _execute(op) is not None


# ─────────────────────────────────────────────
# FETCH FEEDBACK
# ─────────────────────────────────────────────
def fetch_feedback(limit: int = 1000) -> List[Dict[str, Any]]:

    def op(client):
        return (
            client.table("feedback")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

    return _execute(op) or []


# ─────────────────────────────────────────────
# GENERIC SELECT
# ─────────────────────────────────────────────
def fetch_table(
    table: str,
    limit: int = 100,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:

    def op(client):
        query = client.table(table).select("*").limit(limit)

        if filters:
            for k, v in filters.items():
                query = query.eq(k, v)

        return query.execute()

    return _execute(op) or []


# ─────────────────────────────────────────────
# RPC CALL
# ─────────────────────────────────────────────
def call_rpc(function_name: str, payload: Dict[str, Any]):

    def op(client):
        return client.rpc(function_name, payload).execute()

    return _execute(op)


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────
def check_connection() -> bool:
    try:
        client = _get_client()
        if not client:
            return False

        client.table("feedback").select("id").limit(1).execute()
        return True

    except Exception:
        return False