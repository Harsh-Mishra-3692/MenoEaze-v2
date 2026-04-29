# redis_client.py — ELITE (PRODUCTION-READY REDIS CLIENT)

import os
import json
import logging
from typing import Optional, Any

from dotenv import load_dotenv

try:
    import redis
except ImportError:
    raise ImportError("redis not installed. Run: pip install redis")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

DEFAULT_TTL = 3600  # 1 hour

logger = logging.getLogger("menoeaze.redis")
logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────
# CLIENT INIT
# ─────────────────────────────────────────────
class RedisClient:
    def __init__(self, url: str = REDIS_URL):
        self.url = url
        self.client: Optional[redis.Redis] = None
        self._connect()

    def _connect(self):
        try:
            self.client = redis.from_url(
                self.url,
                decode_responses=True,
                socket_timeout=2,
                socket_connect_timeout=2,
            )
            self.client.ping()
            logger.info("[Redis] Connected successfully")

        except Exception as e:
            logger.error(f"[Redis] Connection failed: {e}")
            self.client = None

    # ─────────────────────────────────────────
    # INTERNAL SAFETY CHECK
    # ─────────────────────────────────────────
    def _safe(self):
        return self.client is not None

    # ─────────────────────────────────────────
    # SET VALUE
    # ─────────────────────────────────────────
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = DEFAULT_TTL
    ) -> bool:
        if not self._safe():
            return False

        try:
            serialized = json.dumps(value)
            self.client.set(name=key, value=serialized, ex=ttl)
            return True

        except Exception as e:
            logger.error(f"[Redis] SET failed: {e}")
            return False

    # ─────────────────────────────────────────
    # GET VALUE
    # ─────────────────────────────────────────
    def get(self, key: str) -> Optional[Any]:
        if not self._safe():
            return None

        try:
            val = self.client.get(key)
            if val is None:
                return None
            return json.loads(val)

        except Exception as e:
            logger.error(f"[Redis] GET failed: {e}")
            return None

    # ─────────────────────────────────────────
    # DELETE KEY
    # ─────────────────────────────────────────
    def delete(self, key: str) -> bool:
        if not self._safe():
            return False

        try:
            self.client.delete(key)
            return True

        except Exception as e:
            logger.error(f"[Redis] DELETE failed: {e}")
            return False

    # ─────────────────────────────────────────
    # EXISTS
    # ─────────────────────────────────────────
    def exists(self, key: str) -> bool:
        if not self._safe():
            return False

        try:
            return bool(self.client.exists(key))
        except Exception:
            return False

    # ─────────────────────────────────────────
    # INCREMENT (ATOMIC)
    # ─────────────────────────────────────────
    def incr(self, key: str) -> Optional[int]:
        if not self._safe():
            return None

        try:
            return self.client.incr(key)
        except Exception as e:
            logger.error(f"[Redis] INCR failed: {e}")
            return None

    # ─────────────────────────────────────────
    # HEALTH CHECK
    # ─────────────────────────────────────────
    def health(self) -> bool:
        if not self._safe():
            return False

        try:
            self.client.ping()
            return True
        except Exception:
            return False


# ─────────────────────────────────────────────
# GLOBAL INSTANCE
# ─────────────────────────────────────────────
redis_client = RedisClient()
