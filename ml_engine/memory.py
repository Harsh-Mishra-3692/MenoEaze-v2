# memory.py — ELITE (USER MEMORY MANAGEMENT SYSTEM)

import time
import logging
from typing import Dict, List, Optional
from collections import deque
from threading import Lock

logger = logging.getLogger("menoeaze.memory")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MAX_USERS = 5000
MAX_HISTORY_PER_USER = 20
TTL_SECONDS = 3600  # 1 hour inactivity expiry

# ─────────────────────────────────────────────
# DATA STRUCTURE
# ─────────────────────────────────────────────
class UserMemory:
    """
    Stores recent interactions for a single user.
    """

    def __init__(self):
        self.history = deque(maxlen=MAX_HISTORY_PER_USER)
        self.last_updated = time.time()

    def add(self, query: str, metadata: Optional[dict] = None):
        self.history.append({
            "query": query,
            "metadata": metadata or {},
            "ts": time.time()
        })
        self.last_updated = time.time()

    def get_recent(self, k: int = 5) -> List[dict]:
        return list(self.history)[-k:]

    def is_expired(self) -> bool:
        return (time.time() - self.last_updated) > TTL_SECONDS


# ─────────────────────────────────────────────
# MEMORY STORE (THREAD SAFE)
# ─────────────────────────────────────────────
class MemoryStore:
    """
    Thread-safe in-memory user store.
    """

    def __init__(self):
        self.store: Dict[str, UserMemory] = {}
        self.lock = Lock()

    # ─────────────────────────────────────────
    # ADD INTERACTION
    # ─────────────────────────────────────────
    def add(
        self,
        user_id: str,
        query: str,
        metadata: Optional[dict] = None
    ):
        if not user_id:
            return

        with self.lock:
            if user_id not in self.store:
                if len(self.store) >= MAX_USERS:
                    self._evict_oldest()

                self.store[user_id] = UserMemory()

            self.store[user_id].add(query, metadata)

    # ─────────────────────────────────────────
    # GET MEMORY
    # ─────────────────────────────────────────
    def get(self, user_id: str, k: int = 5) -> List[dict]:
        with self.lock:
            user_mem = self.store.get(user_id)
            if not user_mem:
                return []

            return user_mem.get_recent(k)

    # ─────────────────────────────────────────
    # DELETE USER
    # ─────────────────────────────────────────
    def delete(self, user_id: str):
        with self.lock:
            self.store.pop(user_id, None)

    # ─────────────────────────────────────────
    # CLEANUP (EXPIRED USERS)
    # ─────────────────────────────────────────
    def cleanup(self):
        with self.lock:
            expired = [
                uid for uid, mem in self.store.items()
                if mem.is_expired()
            ]

            for uid in expired:
                del self.store[uid]

            if expired:
                logger.info(f"[Memory] Cleaned {len(expired)} expired users")

    # ─────────────────────────────────────────
    # EVICTION POLICY (LRU APPROX)
    # ─────────────────────────────────────────
    def _evict_oldest(self):
        oldest_uid = None
        oldest_time = float("inf")

        for uid, mem in self.store.items():
            if mem.last_updated < oldest_time:
                oldest_time = mem.last_updated
                oldest_uid = uid

        if oldest_uid:
            del self.store[oldest_uid]
            logger.warning(f"[Memory] Evicted user: {oldest_uid}")


# ─────────────────────────────────────────────
# GLOBAL INSTANCE
# ─────────────────────────────────────────────
memory_store = MemoryStore()

# ─────────────────────────────────────────────
# HELPER FUNCTIONS (API FRIENDLY)
# ─────────────────────────────────────────────
def add_interaction(user_id: str, query: str, metadata: Optional[dict] = None):
    memory_store.add(user_id, query, metadata)


def get_recent_history(user_id: str, k: int = 5) -> List[dict]:
    return memory_store.get(user_id, k)


def delete_user(user_id: str):
    memory_store.delete(user_id)


def cleanup_memory():
    memory_store.cleanup()
