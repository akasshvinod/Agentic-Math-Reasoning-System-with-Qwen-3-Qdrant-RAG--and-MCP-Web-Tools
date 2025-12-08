from __future__ import annotations

import logging
from typing import Dict, Any

from src.graph.state import MathState

logger = logging.getLogger(__name__)


async def router_node(state: MathState) -> MathState:
    """
    Router Node.

    Responsibilities:
      - If input was unsafe → graph will later short-circuit to END.
      - Mark the query as math (for now) so the math pipeline always runs.
      - Initialize shared-state fields so downstream nodes never crash.
    """

    # 1) If guardrail marked input as unsafe, just pass state through.
    if not state.get("is_safe", False):
        logger.warning("[ROUTER] Input unsafe → graph will terminate after this branch.")
        # Initialize required keys with safe defaults
        state.setdefault("broken_queries", [])
        state.setdefault("local_results", [])
        state.setdefault("web_results", [])
        state.setdefault("reranked_results", [])
        state.setdefault("reasoning_output", "")
        state.setdefault("verification", "")
        state.setdefault("needs_web_fallback", False)
        state.setdefault("feedback", None)
        state.setdefault("loop_count", 0)
        state.setdefault("final_output", "")
        return state

    query = (state.get("query") or "").strip()
    logger.info("[ROUTER] Safe query received: %s", query[:120])

    # 2) Initialize missing fields safely (happy path)
    state.setdefault("broken_queries", [])
    state.setdefault("local_results", [])
    state.setdefault("web_results", [])
    state.setdefault("reranked_results", [])
    state.setdefault("reasoning_output", "")
    state.setdefault("verification", "")
    state.setdefault("needs_web_fallback", False)
    state.setdefault("feedback", None)
    state.setdefault("loop_count", 0)
    state.setdefault("final_output", "")

    # 3) For now, treat everything as math so aptitude / word problems also run.
    state["is_math"] = True

    # Optional: log multi-part hints for query_breaker
    lowered = query.lower()
    multi_keywords = [
        " and ", " also ", " then ", " next ",
        " first ", " second ", " third ",
    ]
    if any(k in lowered for k in multi_keywords):
        logger.info("[ROUTER] Multi-part query detected → query_breaker may split it.")
    else:
        logger.info("[ROUTER] Single-part query → query_breaker will likely pass unchanged.")

    return state
