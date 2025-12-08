from __future__ import annotations

import logging
from typing import Dict, Any, List

from langchain_core.documents import Document
from src.graph.state import MathState
from src.hitl.feedback_store import log_feedback
from src.hitl.dspy_evaluator import get_dspy_evaluator  # DSPy-based scoring

logger = logging.getLogger(__name__)


async def feedback_node(state: MathState) -> MathState:
    """
    Human-in-the-loop (HITL) node.

    Responsibilities:
      - Read human feedback from state["feedback"] (optional).
      - Run DSPy evaluator on (query, final_output, retrieval, feedback).
      - Persist a structured record for later training / analysis.
      - Do NOT modify the user-facing answer in this node.

    Expected inputs in state:
      - "query": str
      - "final_output": str
      - "feedback": str (may be empty / missing)
      - "reranked_results": List[Document] (optional, for retrieval context)

    Output in state:
      - state unchanged; side effect is logging to feedback store.
    """

    query: str = (state.get("query") or "").strip()
    final_output: str = (state.get("final_output") or "").strip()
    feedback: str = (state.get("feedback") or "").strip()
    docs: List[Document] = state.get("reranked_results") or []

    if not final_output:
        logger.warning("[HITL] Skipping feedback logging; missing final_output.")
        return state

    # Build compact retrieval context
    context_chunks = []
    for i, d in enumerate(docs[:5], start=1):
        try:
            page = (getattr(d, "page_content", "") or "").strip()
        except Exception:
            page = ""
        if not page:
            continue
        context_chunks.append(f"[Doc {i}] {page[:400]}")
    retrieval_context = "\n\n".join(context_chunks)

    # Run DSPy evaluator
    dspy_report: Any = None
    try:
        evaluator = get_dspy_evaluator()
        dspy_report = evaluator.evaluate(
            user_query=query,
            output_text=final_output,
            retrieved_docs=retrieval_context,
            human_feedback=feedback,
        )
    except Exception as exc:
        logger.exception("[HITL] DSPy evaluator failed: %s", exc)

    # Structure log entry
    entry: Dict[str, Any] = {
        "query": query,
        "final_answer": final_output,
        "feedback": feedback,
        "retrieval_context": retrieval_context,
        "dspy_eval": dspy_report,
    }

    # Persist
    try:
        log_feedback(entry)
        logger.info("[HITL] Logged feedback and DSPy evaluation.")
    except Exception as exc:
        logger.exception("[HITL] Failed to log feedback entry: %s", exc)

    return state
