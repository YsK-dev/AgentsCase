# src/logger.py — Structured persistent logging for the multi-agent system
import json
import logging
import os
from datetime import datetime, timezone

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "agent_system.log")


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line for easy parsing."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            entry["data"] = record.extra_data
        return json.dumps(entry, default=str)


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes structured JSON lines to logs/agent_system.log
    and human-readable output to the console."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # ── File handler (JSON lines, persistent) ───────────────────
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(JSONFormatter())
    logger.addHandler(fh)

    # ── Console handler (human-readable, INFO+) ─────────────────
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(ch)

    return logger


def log_event(logger: logging.Logger, message: str, **kwargs):
    """Convenience: log an INFO message with structured extra data."""
    record = logger.makeRecord(
        logger.name, logging.INFO, "(log_event)", 0, message, (), None
    )
    record.extra_data = kwargs  # type: ignore[attr-defined]
    logger.handle(record)
