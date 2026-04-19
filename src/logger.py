"""Structured logging setup with JSON output and rotating file handler."""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

from pythonjsonlogger.json import JsonFormatter as _JsonFormatter


_initialized = False


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    log_max_bytes: int = 50 * 1024 * 1024,
    log_backup_count: int = 7,
    app_name: str = "beetrade",
) -> logging.Logger:
    """Initialize structured logging with console + rotating file output.

    Returns the root application logger.
    """
    global _initialized
    if _initialized:
        return logging.getLogger(app_name)

    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(app_name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False

    json_fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"

    # Console handler -- human-readable
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler -- JSON lines for machine parsing
    log_file = os.path.join(log_dir, f"{app_name}.jsonl")
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=log_max_bytes,
        backupCount=log_backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = _JsonFormatter(
        json_fmt,
        datefmt="%Y-%m-%dT%H:%M:%S",
        rename_fields={"asctime": "timestamp", "levelname": "level"},
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    _initialized = True
    logger.info("Logging initialized", extra={"log_dir": log_dir, "level": log_level})
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a child logger under the beetrade namespace."""
    base = "beetrade"
    if name:
        return logging.getLogger(f"{base}.{name}")
    return logging.getLogger(base)
