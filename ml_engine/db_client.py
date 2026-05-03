# db_client.py — FINAL ELITE v3 (SAFE + TRUSTED + NON-CORRUPTING)

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
MAX_PAYLOAD_SIZE = 1000

# 🔒 PERSONALIZATION SAFETY LIMITS
MAX_WEIGHT_DELTA = 0.15
WEIGHT_CLAMP_MIN = -1.0
WEIGHT_CLAMP_MAX = 1.0

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
    return hashlib.sha256(str(payload).encode()).hexdigest()


def _validate_identifier(name: str):
    if not isinstance(name, str):
        return None
    name = name.strip()
    if not name.replace("_", "").isalnum():
        return None
    return name


def _clamp_weight(v):
    try:
        return max(WEIGHT_CLAMP_MIN, min(WEIGHT_CLAMP_MAX, float(v)))
    except:
        return 0.0


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
# EXECUTION
# ─────────────────────────────────────────────
def _execute(fn: Callable, op: str):

    if not _check_circuit():
        logger.error(f"[DB][{op}] circuit open")
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
# 🔥 CRITICAL: SAFE USER WEIGHT UPDATE
# ─────────────────────────────────────────────
def update_user_weights(
    user_id: str,
    new_weights: Dict[str, float],
    trust_score: float = 0.5
) -> bool:
    """
    Safe personalization update.

    DOES NOT affect GRU weights.
    Only updates user-level adaptation layer.
    """

    if not _valid_user(user_id):
        return False

    if not isinstance(new_weights, dict):
        return False

    client = get_client()
    if not client:
        return False

    # 🚨 TRUST GATING (ANTI-CORRUPTION)
    trust_score = _safe_float(trust_score)
    if trust_score < 0.2:
        logger.warning("[DB][WEIGHTS] low trust input rejected")
        return False

    try:
        # fetch existing
        existing = _execute(
            lambda: client.table("user_weights")
            .select("*")
            .eq("user_id", user_id)
            .limit(1)
            .execute(),
            "fetch_weights"
        )

        current_weights = {}
        if existing and getattr(existing, "data", None):
            current_weights = existing.data[0].get("weights", {})

        # 🧠 SAFE UPDATE LOGIC
        updated = {}

        for k, v in new_weights.items():

            k = _safe_str(k, 50)
            if not k:
                continue

            v = _clamp_weight(v)

            old = _clamp_weight(current_weights.get(k, 0.0))

            # 🔒 LIMIT CHANGE RATE
            delta = v - old
            if abs(delta) > MAX_WEIGHT_DELTA:
                delta = MAX_WEIGHT_DELTA if delta > 0 else -MAX_WEIGHT_DELTA

            updated[k] = _clamp_weight(old + delta * trust_score)

        payload = {
            "user_id": user_id,
            "weights": updated,
            "updated_at": int(time.time())
        }

        # 🔁 IDEMPOTENT UPSERT
        res = _execute(
            lambda: client.table("user_weights")
            .upsert(payload, on_conflict="user_id")
            .execute(),
            "update_weights"
        )

        return bool(res and getattr(res, "data", None))

    except Exception as e:
        logger.error(f"[DB][WEIGHTS][FAIL] {e}")
        return False


# ─────────────────────────────────────────────
# DOMAIN METHODS (UNCHANGED, SAFE)
# ─────────────────────────────────────────────
def get_user_weights(user_id: str) -> Optional[Dict]:
    if not _valid_user(user_id):
        return None

    client = get_client()
    if not client:
        return None

    res = _execute(
        lambda: client.table("user_weights")
        .select("*")
        .eq("user_id", user_id)
        .limit(1)
        .execute(),
        "get_user_weights"
    )

    if res and hasattr(res, "data") and res.data:
        return res.data[0].get("weights")

    return None