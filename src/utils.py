"""
src/utils.py
Shared utility helpers: config loader, logger factory, directory creator.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load YAML config and return as a plain Python dict."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

_FMT = "%(asctime)s  %(levelname)-8s  %(name)s – %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        # Force UTF-8 on Windows to avoid cp1252 encode errors
        if hasattr(handler.stream, "reconfigure"):
            try:
                handler.stream.reconfigure(encoding="utf-8")
            except Exception:
                pass
        handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Directory helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dirs(*dirs: str | Path) -> None:
    """Create directories (including parents) if they don't already exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def project_root() -> Path:
    """Return the project root (two levels above this file)."""
    return Path(__file__).resolve().parent.parent
