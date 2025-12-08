from __future__ import annotations

import json
import logging
from typing import Dict, List, Any

from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY, GROQ_MODEL  # e.g. llama-3.1-8b-instant

logger = logging.getLogger(__name__)


def _heuristic_split(query: str) -> List[str]:
    """
    Fast rule-based splitter for simple math queries.
    Never splits mathematical expressions, only obvious connective words.
    """
    lowered = query.lower()

    triggers = [" and ", " then ", " also ", " next ", " first ", " second "]

    for trig in triggers:
        if trig in lowered:
            parts = [p.strip() for p in query.split(trig) if p.strip()]
            return parts

    # No heuristic match → single query
    return [query]


def _get_llm() -> ChatGroq | None:
    """Instantiate ChatGroq if GROQ_API_KEY is available; otherwise return None."""
    if not GROQ_API_KEY:
        logger.warning("[QUERY BREAKER] GROQ_API_KEY not set → skipping LLM fallback.")
        return None

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.0,
        max_retries=2,
    )
    return llm


async def _llm_split(query: str) -> List[str]:
    """
    Fallback LLM-based splitter using Groq.

    The model is instructed to return ONLY a JSON list of strings,
    e.g. ["subquestion 1", "subquestion 2"].
    """
    llm = _get_llm()
    if llm is None:
        return [query]

    system_msg = (
        "You are a math query router.\n"
        "Given a user math question, break it into the minimal number of "
        "independent sub-questions, or return a single-item list if it is atomic.\n"
        "Output MUST be valid JSON: a list of strings, nothing else."
    )

    user_msg = f'Query: "{query}"'

    try:
        ai_msg = await llm.ainvoke(
            [
                ("system", system_msg),
                ("user", user_msg),
            ]
        )
        content = ai_msg.content or ""

        subqs = json.loads(content)
        if isinstance(subqs, list) and all(isinstance(x, str) for x in subqs):
            return [s.strip() for s in subqs if s.strip()]

        logger.warning(
            "[QUERY BREAKER] LLM returned non-list or invalid items; content=%r", content
        )
    except Exception as exc:
        logger.exception("[QUERY BREAKER] LLM fallback failed: %s", exc)

    # Fallback to single query if anything goes wrong
    return [query]


async def query_breaker_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 3: Query Breaker.

    Splits multi-part math questions into smaller ones.

    Logic:
      1) Use lightweight heuristics for obvious multi-step queries.
      2) If heuristics produce a single part, optionally call Groq LLM
         to suggest a JSON list of sub-queries.
      3) Always write a non-empty list to state["broken_queries"].
    """
    query = (state.get("query") or "").strip()
    logger.info("[QUERY BREAKER] Received query: %r", query)

    # 1) Heuristic split
    heuristics = _heuristic_split(query)

    if len(heuristics) > 1:
        logger.info("[QUERY BREAKER] Heuristic split → %s", heuristics)
        state["broken_queries"] = heuristics
        return state

    # 2) LLM fallback (optional)
    logger.info("[QUERY BREAKER] Trying LLM fallback for splitting...")
    subqueries = await _llm_split(query)

    if not subqueries:
        subqueries = [query]

    logger.info("[QUERY BREAKER] Final subqueries: %s", subqueries)
    state["broken_queries"] = subqueries
    return state
