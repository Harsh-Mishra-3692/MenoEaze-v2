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
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("[DB] SUPABASE_URL or SUPABASE_KEY not configured — DB unavailable")
        return None

    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("[DB] Supabase client created")
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
def insert_bulk(table: str, rows: List[Dict[str, Any]], upsert: bool = False, on_conflict: str = "") -> bool:

    if not rows:
        return True

    def op(client):
        query = client.table(table)
        if upsert:
            return query.upsert(rows, on_conflict=on_conflict).execute() if on_conflict else query.upsert(rows).execute()
        return query.insert(rows).execute()

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
    filters: Optional[Dict[str, Any]] = None,
    order_by: Optional[str] = None
) -> List[Dict[str, Any]]:

    def op(client):
        query = client.table(table).select("*").limit(limit)

        if filters:
            for k, v in filters.items():
                query = query.eq(k, v)
        
        if order_by:
            query = query.order(order_by, desc=True)

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

        # Schema-agnostic check: just verify the client can reach Supabase
        # Use symptom_logs as it's guaranteed to exist in the stable schema
        client.table("symptom_logs").select("id").limit(1).execute()
        return True

    except Exception:
        return False

# ─────────────────────────────────────────────
# USER MEMORY
# ─────────────────────────────────────────────
def get_user_memory(user_id: str) -> Dict[str, Any]:
    def op(client):
        res = client.table("user_memory").select("data").eq("user_id", user_id).limit(1).execute()
        if res and hasattr(res, "data") and res.data:
            return {"history": res.data[0].get("data", [])}
        return {}
    return _execute(op) or {}

def update_user_memory(user_id: str, new_entry: Dict[str, Any]) -> bool:
    def op(client):
        # fetch existing
        res = client.table("user_memory").select("data").eq("user_id", user_id).limit(1).execute()
        current_data = []
        if res and hasattr(res, "data") and res.data:
            current_data = res.data[0].get("data", [])
        
        current_data.append(new_entry)
        current_data = current_data[-50:]
        
        res = client.table("user_memory").upsert({
            "user_id": user_id,
            "data": current_data
        }).execute()
        return res
    return _execute(op) is not None