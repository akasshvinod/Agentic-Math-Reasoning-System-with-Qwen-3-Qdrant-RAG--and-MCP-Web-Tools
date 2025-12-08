from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

FEEDBACK_DIR = Path("feedback/logs")
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = FEEDBACK_DIR / "feedback_dataset.jsonl"


def log_feedback(entry: Dict[str, Any]) -> None:
    """
    Append a feedback entry to a JSONL dataset on disk.

    This dataset can later be used for:
      - supervised fine-tuning
      - RLHF-style preference modeling
      - error analysis and evaluation
    """
    try:
        record = dict(entry)  # shallow copy
        record["timestamp"] = datetime.now(timezone.utc).isoformat()

        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("[HITL] Feedback entry appended to %s", LOG_PATH)
    except Exception as exc:
        logger.exception("[HITL] Failed to write feedback entry: %s", exc)
