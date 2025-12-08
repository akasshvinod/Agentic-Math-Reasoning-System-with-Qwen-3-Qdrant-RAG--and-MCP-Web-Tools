# src/utils/logging.py

import logging
import sys
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def setup_logging(level: LogLevel = "INFO") -> None:
    """
    Configure consistent logging across the project.

    Args:
        level: Logging level as string ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
    """
    # Convert string to logging constant, default to INFO if invalid
    log_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(log_level)

    # Remove existing handlers to avoid duplicate logs in notebooks/CLI reloads
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    root.info("Logging initialized at %s level.", level.upper())
