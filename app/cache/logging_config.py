"""Rotating file logging setup, writing into the cache directory's logs folder."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

from app.cache.cache_manager import CacheManager

LOG_FILENAME = "app.log"
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

_configured = False


def configure_logging(cache_manager: Optional[CacheManager] = None, level: int = logging.INFO) -> None:
    """Attach a rotating file handler (5MB x 3) to the root logger. Idempotent."""
    global _configured
    if _configured:
        return

    manager = cache_manager or CacheManager()
    log_path = manager.logs_dir / LOG_FILENAME

    handler = RotatingFileHandler(
        log_path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter(LOG_FORMAT))

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, configuring rotating file logging on first use."""
    configure_logging()
    return logging.getLogger(name)
