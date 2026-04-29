# config.py — ELITE (ROBUST + SAFE + PRODUCTION CONFIG SYSTEM)

import os
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from dotenv import load_dotenv

# ─────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────
logger = logging.getLogger("menoeaze.config")
logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────
# LOAD ENV (CRITICAL FIX)
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
    logger.info(f"[CONFIG] Loaded .env from {ENV_PATH}")
else:
    logger.warning(f"[CONFIG] .env file not found at {ENV_PATH}")


# ─────────────────────────────────────────────
# TYPE CAST HELPERS
# ─────────────────────────────────────────────
def _to_bool(val: str) -> bool:
    return str(val).lower() in ("true", "1", "yes")


def _safe_cast(val: Any, cast: Callable, key: str):
    try:
        return cast(val)
    except Exception:
        raise ValueError(f"[CONFIG] Failed to cast '{key}' with value '{val}'")


# ─────────────────────────────────────────────
# ENV FETCH
# ─────────────────────────────────────────────
def _get_env(
    key: str,
    default: Optional[Any] = None,
    required: bool = False,
    cast: Optional[Callable] = None
):
    val = os.getenv(key)

    # ── Required check ───────────────────────
    if val is None:
        if required:
            raise RuntimeError(f"[CONFIG] Missing required env variable: {key}")
        val = default

    # ── Cast if needed ──────────────────────
    if val is not None and cast:
        val = _safe_cast(val, cast, key)

    return val


# ─────────────────────────────────────────────
# CONFIG STRUCTS
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class AppConfig:
    env: str
    debug: bool
    log_level: str


@dataclass(frozen=True)
class ModelConfig:
    seq_len: int
    features: int
    adapt_threshold: int


@dataclass(frozen=True)
class RAGConfig:
    top_k: int
    similarity_threshold: float
    max_context_chars: int


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    temperature: float
    max_tokens: int
    api_key: Optional[str]


@dataclass(frozen=True)
class RedisConfig:
    url: str
    ttl: int


@dataclass(frozen=True)
class DBConfig:
    url: str
    key: str


# ─────────────────────────────────────────────
# LOAD CONFIG
# ─────────────────────────────────────────────
def load_config():
    try:
        config = {
            "app": AppConfig(
                env=_get_env("APP_ENV", "dev"),
                debug=_get_env("DEBUG", "false", cast=_to_bool),
                log_level=_get_env("LOG_LEVEL", "INFO"),
            ),

            "model": ModelConfig(
                seq_len=_get_env("SEQ_LEN", 5, cast=int),
                features=_get_env("FEATURES", 11, cast=int),
                adapt_threshold=_get_env("ADAPT_THRESHOLD", 15, cast=int),
            ),

            "rag": RAGConfig(
                top_k=_get_env("RAG_TOP_K", 8, cast=int),
                similarity_threshold=_get_env("RAG_SIM_THRESHOLD", 0.65, cast=float),
                max_context_chars=_get_env("RAG_MAX_CONTEXT", 3500, cast=int),
            ),

            "llm": LLMConfig(
                provider=_get_env("LLM_PROVIDER", "groq"),
                model=_get_env("LLM_MODEL", "llama3-8b-8192"),
                temperature=_get_env("LLM_TEMP", 0.2, cast=float),
                max_tokens=_get_env("LLM_MAX_TOKENS", 400, cast=int),
                api_key=_get_env("GROQ_API_KEY", required=False),
            ),

            "redis": RedisConfig(
                url=_get_env("REDIS_URL", "redis://localhost:6379/0"),
                ttl=_get_env("REDIS_TTL", 3600, cast=int),
            ),

            "db": DBConfig(
                url=_get_env("SUPABASE_URL", required=True),
                key=_get_env("SUPABASE_KEY", required=True),
            ),
        }

        logger.info(f"[CONFIG] Environment: {config['app'].env}")
        logger.info(f"[CONFIG] Debug mode: {config['app'].debug}")

        return config

    except Exception as e:
        logger.error(f"[CONFIG] Failed to load configuration: {e}")
        raise


# ─────────────────────────────────────────────
# GLOBAL INSTANCE
# ─────────────────────────────────────────────
CONFIG = load_config()