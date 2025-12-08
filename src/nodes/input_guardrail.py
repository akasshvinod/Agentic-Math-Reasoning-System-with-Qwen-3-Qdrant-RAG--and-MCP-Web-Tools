from __future__ import annotations

import logging
from typing import Dict, Any

from src.graph.state import MathState

logger = logging.getLogger(__name__)


def input_guardrail_node(state: MathState) -> MathState:
    """
    Node 1: Input Guardrail (AI Gateway).

    Ensures:
      - Query is non-empty.
      - Query is free of obviously unsafe content.
      - Query is plausibly mathematics-related.

    Sets:
      state["is_safe"]: bool
      state["reasoning_output"]: str (only if unsafe / rejected)
    """
    raw_query = state.get("query", "")
    query = (raw_query or "").strip()
    logger.info("[INPUT GUARDRAIL] Received query: %r", query)

    # Default
    state.setdefault("is_safe", False)

    # 1) Empty or invalid input
    if not query:
        logger.warning("[INPUT GUARDRAIL] Empty query.")
        state["is_safe"] = False
        state["reasoning_output"] = "Invalid query: input is empty."
        return state

    lowered = query.lower()

    # 2) Block clearly harmful / unsafe content (very simple keyword gate)
    harmful_keywords = [
        "kill", "bomb", "attack", "hack", "suicide",
        "nsfw", "virus", "exploit",
    ]
    if any(word in lowered for word in harmful_keywords):
        logger.warning("[INPUT GUARDRAIL] Unsafe content detected.")
        state["is_safe"] = False
        state["reasoning_output"] = (
            "Your query appears to contain unsafe or harmful content. "
            "This math agent only answers mathematics questions."
        )
        return state

    # 3) Ensure the query is mathematics-focused (heuristic)
    math_keywords = [
        "solve", "equation", "calculate", "find", "how many", "value",
        "integral", "derivative", "number", "arithmatic", "angle", "axis",
        "center", "clock", "compare", "count", "total",
        "limit", "probability", "algebra", "geometry", "matrix",
        "expression", "formula", "constant", "variable", "trigonometric",
        "circle", "triangle", "degree","percentage","sin","cos","tan","lim"
        "compute", "evaluate", "estimate", "simplify", "function", "theorem",
        "equals", "root", "square", "linear", "polynomial", "area", "bar",
        "factor", "time", "times","profit","rule","power","solution",
        "prove", "proof", "measure", "series", "sum", "product", "ratio",
        "vector", "graph", "algorithm", "set", "digit", "fraction", "unit",
        "x^", "y^", "z^",
    ]

    if not any(k in lowered for k in math_keywords):
        logger.warning("[INPUT GUARDRAIL] Non-math query rejected.")
        state["is_safe"] = False
        state["reasoning_output"] = (
            "This agent only supports mathematics-related questions. "
            "Please ask a math question (e.g., an equation, proof, or calculation)."
        )
        return state

    # 4) Passed guardrails → allow continuation
    logger.info("[INPUT GUARDRAIL] Query accepted as safe math query.")
    state["is_safe"] = True
    return state
