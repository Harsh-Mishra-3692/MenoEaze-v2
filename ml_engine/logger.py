# logger.py — FULL OBSERVABILITY SYSTEM (MENOEAZE)

import logging
import sys
import json
import traceback
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from contextvars import ContextVar

from ml_engine.config import CONFIG

# ─────────────────────────────────────────────
# CONTEXT (ASYNC SAFE)
# ─────────────────────────────────────────────
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
user_id_ctx: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
trace_id_ctx: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
LOG_LEVEL = CONFIG["app"].log_level.upper()
ENV = CONFIG["app"].env.lower()

MAX_LOG_SIZE = 2000
LOG_SAMPLE_RATE = 1.0 if ENV != "production" else 0.2  # sampling in prod


# ─────────────────────────────────────────────
# UTIL
# ─────────────────────────────────────────────
def _now():
    return datetime.utcnow().isoformat()


def _safe_serialize(obj):
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return str(obj)[:500]


def _should_log() -> bool:
    if LOG_SAMPLE_RATE >= 1.0:
        return True
    return (time.time() % 1) < LOG_SAMPLE_RATE


# ─────────────────────────────────────────────
# FORMATTERS
# ─────────────────────────────────────────────
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log = {
            "ts": _now(),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", "log"),
            "msg": record.getMessage(),
        }

        # context
        if request_id_ctx.get():
            log["request_id"] = request_id_ctx.get()
        if user_id_ctx.get():
            log["user_id"] = user_id_ctx.get()
        if trace_id_ctx.get():
            log["trace_id"] = trace_id_ctx.get()

        # structured data
        if hasattr(record, "extra_data") and record.extra_data:
            log["extra"] = record.extra_data

        # exception
        if record.exc_info:
            log["error"] = {
                "type": record.exc_info[0].__name__,
                "msg": str(record.exc_info[1]),
                "trace": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log, default=str)


class DevFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        prefix = f"[{record.levelname}] {record.name}"

        if request_id_ctx.get():
            prefix += f" [req:{request_id_ctx.get()}]"
        if user_id_ctx.get():
            prefix += f" [user:{user_id_ctx.get()}]"

        return f"{prefix} | {record.getMessage()}"


# ─────────────────────────────────────────────
# HANDLER
# ─────────────────────────────────────────────
def _get_handler():
    handler = logging.StreamHandler(sys.stdout)

    if ENV == "production":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(DevFormatter())

    return handler


# ─────────────────────────────────────────────
# LOGGER FACTORY
# ─────────────────────────────────────────────
def get_logger(name: str):
    logger = logging.getLogger(name)

    if getattr(logger, "_initialized", False):
        return logger

    logger.setLevel(LOG_LEVEL)
    logger.addHandler(_get_handler())
    logger.propagate = False
    logger._initialized = True

    return logger


# ─────────────────────────────────────────────
# ROOT INIT
# ─────────────────────────────────────────────
def init_logging():
    root = logging.getLogger()

    if getattr(root, "_initialized", False):
        return

    root.setLevel(LOG_LEVEL)
    root.addHandler(_get_handler())
    root._initialized = True


# ─────────────────────────────────────────────
# CONTEXT MANAGEMENT
# ─────────────────────────────────────────────
def start_request(user_id: Optional[str] = None):
    request_id = str(uuid.uuid4())
    trace_id = str(uuid.uuid4())

    request_id_ctx.set(request_id)
    trace_id_ctx.set(trace_id)

    if user_id:
        user_id_ctx.set(user_id)

    return request_id


def end_request():
    request_id_ctx.set(None)
    user_id_ctx.set(None)
    trace_id_ctx.set(None)


# ─────────────────────────────────────────────
# STRUCTURED LOGGING
# ─────────────────────────────────────────────
def log_event(
    logger: logging.Logger,
    message: str,
    level: str = "info",
    event: str = "log",
    extra: Optional[Dict[str, Any]] = None,
    exc: Optional[Exception] = None,
):
    if not _should_log():
        return

    try:
        extra_clean = {}

        if extra:
            for k, v in extra.items():
                if isinstance(v, (dict, list, str, int, float, bool)):
                    extra_clean[k] = v
                else:
                    extra_clean[k] = str(v)

        if extra_clean:
            serialized = _safe_serialize(extra_clean)
            if len(serialized) > MAX_LOG_SIZE:
                extra_clean = {"truncated": True}

        payload = {
            "extra_data": extra_clean,
            "event": event
        }

        level = level.lower()

        if level == "debug":
            logger.debug(message, extra=payload)
        elif level == "warning":
            logger.warning(message, extra=payload)
        elif level == "error":
            logger.error(message, exc_info=exc, extra=payload)
        elif level == "critical":
            logger.critical(message, exc_info=exc, extra=payload)
        else:
            logger.info(message, extra=payload)

    except Exception:
        pass


# ─────────────────────────────────────────────
# TIMER (PIPELINE TRACING)
# ─────────────────────────────────────────────
class Timer:
    def __init__(self, logger, label: str):
        self.logger = logger
        self.label = label
        self.start = time.time()

    def stop(self, extra: Optional[Dict] = None):
        duration = (time.time() - self.start) * 1000

        log_event(
            self.logger,
            f"{self.label} completed",
            event="latency",
            extra={
                "latency_ms": round(duration, 2),
                **(extra or {})
            }
        )


# ─────────────────────────────────────────────
# TRACE HELPER (VERY POWERFUL)
# ─────────────────────────────────────────────
def trace_step(logger, step_name: str, extra: Optional[Dict] = None):
    log_event(
        logger,
        f"{step_name}",
        event="pipeline_step",
        extra=extra
    )