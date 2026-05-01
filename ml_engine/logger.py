# logger.py — ELITE CENTRALIZED LOGGING SYSTEM (PRODUCTION-GRADE)

import logging
import sys
import json
from datetime import datetime
from typing import Optional, Dict, Any
from contextvars import ContextVar

from config import CONFIG

# ─────────────────────────────────────────────
# CONTEXT (for request tracing, user_id, etc.)
# ─────────────────────────────────────────────
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
LOG_LEVEL = CONFIG["app"].log_level.upper()
ENV = CONFIG["app"].env.lower()

# ─────────────────────────────────────────────
# FORMATTERS
# ─────────────────────────────────────────────
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # request context
        request_id = request_id_ctx.get()
        if request_id:
            log_record["request_id"] = request_id

        # structured extra
        if hasattr(record, "extra_data") and record.extra_data:
            log_record["extra"] = record.extra_data

        # exception handling
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record, default=str)


class DevFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        request_id = request_id_ctx.get()
        prefix = f"[{record.levelname}] {record.name}"

        if request_id:
            prefix += f" [req:{request_id}]"

        return f"{prefix} | {record.getMessage()}"


# ─────────────────────────────────────────────
# HANDLER SETUP (SAFE)
# ─────────────────────────────────────────────
def _get_handler() -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)

    if ENV == "production":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(DevFormatter())

    return handler


# ─────────────────────────────────────────────
# LOGGER FACTORY (SAFE, NO DUPLICATES)
# ─────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if getattr(logger, "_initialized", False):
        return logger

    logger.setLevel(LOG_LEVEL)
    logger.addHandler(_get_handler())
    logger.propagate = False

    logger._initialized = True  # custom flag

    return logger


# ─────────────────────────────────────────────
# ROOT LOGGER INIT (CALL ON STARTUP)
# ─────────────────────────────────────────────
def init_logging():
    root = logging.getLogger()

    if getattr(root, "_initialized", False):
        return

    root.setLevel(LOG_LEVEL)
    root.addHandler(_get_handler())

    root._initialized = True


# ─────────────────────────────────────────────
# CONTEXT HELPERS
# ─────────────────────────────────────────────
def set_request_id(request_id: str):
    request_id_ctx.set(request_id)


def clear_request_id():
    request_id_ctx.set(None)


# ─────────────────────────────────────────────
# STRUCTURED LOGGING HELPER
# ─────────────────────────────────────────────
def log_event(
    logger: logging.Logger,
    message: str,
    level: str = "info",
    extra: Optional[Dict[str, Any]] = None,
    exc: Optional[Exception] = None,
):
    """
    Structured logging wrapper with safe metadata handling
    """

    extra_data = {"extra_data": extra} if extra else {}

    level = level.lower()

    if level == "debug":
        logger.debug(message, extra=extra_data)
    elif level == "warning":
        logger.warning(message, extra=extra_data)
    elif level == "error":
        logger.error(message, exc_info=exc, extra=extra_data)
    elif level == "critical":
        logger.critical(message, exc_info=exc, extra=extra_data)
    else:
        logger.info(message, extra=extra_data)