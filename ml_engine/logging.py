# logging.py — ELITE (CENTRALIZED LOGGING SYSTEM)

import logging
import sys
import json
from datetime import datetime
from typing import Optional

from ml_engine.config import CONFIG

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
LOG_LEVEL = CONFIG["app"].log_level.upper()
ENV = CONFIG["app"].env

# ─────────────────────────────────────────────
# FORMATTERS
# ─────────────────────────────────────────────
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        # optional fields
        if hasattr(record, "extra_data"):
            log_record["extra"] = record.extra_data

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record)


class DevFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return (
            f"[{record.levelname}] "
            f"{record.name} | "
            f"{record.getMessage()}"
        )


# ─────────────────────────────────────────────
# HANDLER SETUP
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
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # avoid duplicate handlers

    logger.setLevel(LOG_LEVEL)

    handler = _get_handler()
    logger.addHandler(handler)

    logger.propagate = False

    return logger


# ─────────────────────────────────────────────
# GLOBAL ROOT LOGGER INIT
# ─────────────────────────────────────────────
def init_logging():
    root = logging.getLogger()

    if root.handlers:
        return

    root.setLevel(LOG_LEVEL)

    handler = _get_handler()
    root.addHandler(handler)


# ─────────────────────────────────────────────
# STRUCTURED LOG HELPERS
# ─────────────────────────────────────────────
def log_event(
    logger: logging.Logger,
    message: str,
    level: str = "info",
    extra: Optional[dict] = None
):
    """
    Structured logging helper
    """

    extra_data = {"extra_data": extra} if extra else {}

    level = level.lower()

    if level == "debug":
        logger.debug(message, extra=extra_data)
    elif level == "warning":
        logger.warning(message, extra=extra_data)
    elif level == "error":
        logger.error(message, extra=extra_data)
    elif level == "critical":
        logger.critical(message, extra=extra_data)
    else:
        logger.info(message, extra=extra_data)
