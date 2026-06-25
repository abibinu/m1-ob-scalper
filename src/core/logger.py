"""
Structured logging setup.

Every module obtains a logger via ``get_logger(__name__)``.
Output goes to:
  - console (INFO+)
  - rotating file  logs/ob_scalper.log  (DEBUG+, max 10 MB × 5 backups)

The log format is JSON-compatible when LOG_JSON=1 is set, otherwise
human-readable for local development.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

# ── constants ────────────────────────────────────────────────────────────────
LOG_DIR = Path(os.environ.get("LOG_DIR", "logs"))
LOG_FILE = LOG_DIR / "ob_scalper.log"
LOG_JSON = os.environ.get("LOG_JSON", "0") == "1"
LOG_LEVEL_CONSOLE = os.environ.get("LOG_LEVEL_CONSOLE", "INFO").upper()
LOG_LEVEL_FILE = os.environ.get("LOG_LEVEL_FILE", "DEBUG").upper()

_INITIALIZED = False


def _build_formatter(json_mode: bool) -> logging.Formatter:
    if json_mode:
        # Simple JSON-ish format — avoids extra deps
        fmt = (
            '{"time":"%(asctime)s","level":"%(levelname)s",'
            '"logger":"%(name)s","msg":%(message)r}'
        )
    else:
        fmt = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
    return logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S")


def _setup_root_logger() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # let handlers filter

    # ── console handler ──────────────────────────────────────────────────────
    console_h = logging.StreamHandler(sys.stdout)
    console_h.setLevel(getattr(logging, LOG_LEVEL_CONSOLE, logging.INFO))
    console_h.setFormatter(_build_formatter(json_mode=False))
    root.addHandler(console_h)

    # ── rotating file handler ────────────────────────────────────────────────
    file_h = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_h.setLevel(getattr(logging, LOG_LEVEL_FILE, logging.DEBUG))
    file_h.setFormatter(_build_formatter(json_mode=LOG_JSON))
    root.addHandler(file_h)

    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, initializing the root logger on first call."""
    _setup_root_logger()
    return logging.getLogger(name)
