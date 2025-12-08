# src/tools/local_rag_tool.py

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List

from langchain_core.tools import tool

from src.embedder.embedder import get_embedder
from src.tools.qdrant_tool import get_qdrant

logger = logging.getLogger(__name__)


@tool
def local_rag_search(
    query: str,
    top_k: int = 5,
    topic: Optional[str] = None,
    difficulty: Optional[str] = None,
    subject: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Local RAG retrieval over the Hendrycks Math index.

    - Embeds the user query with BGE-Large (query mode).
    - Queries Qdrant for top_k nearest neighbors.
    - Optionally filters by topic, difficulty, and subject metadata.
    - If filters yield no results, automatically retries without filters.

    Args:
        query: User math question in natural language.
        top_k: Number of neighbors to return (1–20 recommended).
        topic: Optional metadata filter, e.g. "Algebra", "Precalculus".
        difficulty: Optional difficulty filter, e.g. "Level 2".
        subject: Optional subject filter, e.g. "algebra", "precalculus".

    Returns:
        dict with:
            - query: original query
            - top_k: requested number of results
            - used_filters: filters effectively applied (with fallback info)
            - results: list of {id, score, payload}
    """

    logger.info("[TOOL:RAG] Received query: %s", query)

    cleaned = (query or "").strip()
    if not cleaned:
        return {"error": "Query cannot be empty."}

    if top_k <= 0:
        top_k = 5
    if top_k > 20:
        top_k = 20  # keep latency reasonable inside LangGraph loops

    embedder = get_embedder()
    qdrant = get_qdrant()

    # 1) Embed query in "query" mode
    try:
        q_vec: List[float] = embedder.embed(cleaned, as_query=True)
    except Exception as exc:
        logger.exception("[TOOL:RAG] Failed to embed query: %s", exc)
        return {"error": "Embedding failed."}

    # 2) Build metadata filter dict (only non-empty fields)
    metadata_filter: Dict[str, Any] = {}
    if topic:
        metadata_filter["topic"] = topic
    if difficulty:
        metadata_filter["difficulty"] = difficulty
    if subject:
        metadata_filter["subject"] = subject

    # 3) Primary search (with filters if provided)
    try:
        results = qdrant.search(
            query_vector=q_vec,
            top_k=top_k,
            filter_metadata=metadata_filter or None,
        )
    except Exception as exc:
        logger.exception("[TOOL:RAG] Qdrant search failed: %s", exc)
        return {"error": "Search failed."}

    used_filters: Dict[str, Any]

    # 4) Fallback: if filters yield no hits, retry without filters
    if not results and metadata_filter:
        logger.info(
            "[TOOL:RAG] No results with filters %s → retrying without filters",
            metadata_filter,
        )
        try:
            results = qdrant.search(
                query_vector=q_vec,
                top_k=top_k,
                filter_metadata=None,
            )
        except Exception as exc:
            logger.exception("[TOOL:RAG] Fallback search (no filters) failed: %s", exc)
            return {"error": "Search failed."}

        used_filters = {
            "requested": metadata_filter,
            "effective": {},  # actually used in second search
            "fallback": True,
        }
    else:
        used_filters = {
            "requested": metadata_filter,
            "effective": metadata_filter,
            "fallback": False,
        }

    logger.info(
        "[TOOL:RAG] Retrieved %d results (top_k=%d, fallback=%s, effective_filters=%s)",
        len(results),
        top_k,
        used_filters["fallback"],
        used_filters["effective"],
    )

    return {
        "query": cleaned,
        "top_k": top_k,
        "used_filters": used_filters,
        "results": results,
    }
