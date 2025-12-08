from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional

from langchain_core.documents import Document

from src.tools.local_rag_tool import local_rag_search
from src.config import RAG_THRESHOLD
from src.graph.state import MathState

logger = logging.getLogger(__name__)


async def local_rag_node(state: MathState) -> MathState:
    """
    Node 4: Local RAG Retrieval (Qdrant via Tool).

    Responsibilities:
      - Choose the retrieval query (original or first broken subquery).
      - Call local_rag_search tool to get top-k matches.
      - Convert raw results into LangChain Document objects.
      - Set `needs_web_fallback` based on top score vs RAG_THRESHOLD.
      - Populate `local_results` in the shared state.
    """

    query = (state.get("query") or "").strip()
    broken = state.get("broken_queries") or []

    # Use first subquery if query was split
    retrieval_query = (broken[0] or "").strip() if broken else query

    logger.info("[LOCAL RAG] Searching for: %r", retrieval_query)

    if not retrieval_query:
        logger.warning("[LOCAL RAG] Empty retrieval query → skipping RAG.")
        state["local_results"] = []
        state["needs_web_fallback"] = True
        return state

    # Optional filters
    topic: Optional[str] = state.get("topic")
    difficulty: Optional[str] = state.get("difficulty")
    subject: Optional[str] = state.get("subject")
    top_k: int = int(state.get("rag_top_k", 5)) or 5

    # ------------------------------------------------------------------
    # TOOL CALL
    # ------------------------------------------------------------------
    try:
        tool_output: Dict[str, Any] = local_rag_search.invoke(
            {
                "query": retrieval_query,
                "top_k": top_k,
                "topic": topic,
                "difficulty": difficulty,
                "subject": subject,
            }
        )
    except Exception as exc:
        logger.exception("[LOCAL RAG] Tool invocation failed: %s", exc)
        state["local_results"] = []
        state["needs_web_fallback"] = True
        return state

    raw_results = tool_output.get("results") or []
    logger.info("[LOCAL RAG] Got %d raw matches.", len(raw_results))

    docs: List[Document] = []

    for r in raw_results:
        payload = r.get("payload") or {}
        problem = (payload.get("problem") or "").strip()
        solution = (payload.get("solution") or "").strip()
        score = r.get("score", 0.0)

        if not problem:
            # Skip malformed entries
            continue

        text_parts = [f"Problem: {problem}"]
        if solution:
            text_parts.append(f"Solution: {solution}")
        text = "\n".join(text_parts)

        doc = Document(
            page_content=text,
            metadata={
                "origin": "local_rag",
                "score": float(score),
                "topic": payload.get("topic"),
                "difficulty": payload.get("difficulty"),
                "subject": payload.get("subject"),
                "split": payload.get("source_split"),
            },
        )
        docs.append(doc)

    state["local_results"] = docs

    # ------------------------------------------------------------------
    # Threshold logic
    # ------------------------------------------------------------------
    if not docs:
        logger.warning("[LOCAL RAG] No results → web fallback required.")
        state["needs_web_fallback"] = True
        return state

    # Assume results are sorted by score descending
    top_score = float(docs[0].metadata.get("score", 0.0))

    logger.info(
        "[LOCAL RAG] Top score = %.4f, Threshold = %.4f",
        top_score,
        RAG_THRESHOLD,
    )

    state["needs_web_fallback"] = top_score < RAG_THRESHOLD
    return state
