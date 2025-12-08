from __future__ import annotations

import json
import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

THINK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
MULTI_NL_PATTERN = re.compile(r"\n{3,}")
FORBIDDEN_PATTERNS = [
    r"\bas an AI\b",
    r"\bI am unable to\b",
    r"\bsystem prompt\b",
    r"\bdeveloper message\b",
    r"\bOpenAI policies\b",
    r"\breasoning process\b",
    r"\bchain of thought\b",
]


def _strip_think_tags(text: str) -> str:
    """
    Remove <think>...</think> blocks from LLM outputs (Qwen, DeepSeek, etc.).
    Chain-of-thought must never be shown to the user.
    """
    if not text:
        return ""
    cleaned = THINK_PATTERN.sub("", text)
    return cleaned.strip()


def _sanitize_output(text: str) -> str:
    """
    Additional guardrail layer:
    - Strip meta / system-leaking phrases.
    - Normalize excessive newlines.
    """
    if not text:
        return ""

    for pat in FORBIDDEN_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    # Collapse too many blank lines
    text = MULTI_NL_PATTERN.sub("\n\n", text)

    return text.strip()


def _parse_verifier_json(raw: Any) -> Optional[Dict[str, Any]]:
    """
    Accepts either:
      - dict (already parsed), or
      - JSON string from verifier_node.

    Returns a dict or None on failure.
    """
    if raw is None:
        return None

    if isinstance(raw, dict):
        return raw

    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            logger.warning("[OUTPUT] Failed to parse verifier JSON string.")
            return None

    logger.warning("[OUTPUT] Unexpected verifier payload type: %r", type(raw))
    return None


async def output_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Final node that prepares the answer sent to the user.

    Inputs (state):
      - "reasoning_output": str (from reasoning_node)
      - "verification": JSON string or dict from verifier_node (optional)

    Output (state):
      - "final_output": str (user-facing answer)
    """
    reasoning_output = (state.get("reasoning_output") or "").strip()
    raw_verification = state.get("verification")

    logger.info("[OUTPUT] Formatting final output...")

    # 1) Strip chain-of-thought tags
    cleaned = _strip_think_tags(reasoning_output)

    # 2) Sanitize meta / leakage
    cleaned = _sanitize_output(cleaned)

    # 3) Try to use verifier-improved answer when available
    verification = _parse_verifier_json(raw_verification)
    improved_answer = None
    if verification and isinstance(verification.get("improved_answer"), str):
        candidate = verification["improved_answer"].strip()
        if candidate:
            improved_answer = candidate

    final_answer = improved_answer if improved_answer else cleaned

    # 4) Minimal, consistent wrapper for UI / API
    formatted_output = final_answer.strip()

    state["final_output"] = formatted_output
    return state
