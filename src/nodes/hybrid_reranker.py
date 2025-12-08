from __future__ import annotations

import asyncio
import logging
from typing import Dict, Any, List, Tuple, Iterable

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder  # [web:319][web:328]

from src.graph.state import MathState

logger = logging.getLogger(__name__)

# Cross-encoder configuration
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"  # [web:318]
RERANK_TOP_K = 8
RERANK_BATCH_SIZE = 32


# -----------------------------------------
# Helpers
# -----------------------------------------
def _make_candidate_from_local(doc: Document) -> Tuple[str, Dict[str, Any]]:
    """Convert local RAG Document into (text, metadata)."""
    meta = dict(doc.metadata or {})
    meta.setdefault("origin", "local_rag")
    text = (doc.page_content or "").strip()
    return text, meta


def _make_candidate_from_web(item: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Convert MCP web result dict → (text, metadata)."""
    title = (item.get("title") or "").strip()
    url = (item.get("url") or "").strip()
    content = (item.get("content") or "").strip()

    parts: List[str] = []
    if title:
        parts.append(f"Title: {title}")
    if url:
        parts.append(f"URL: {url}")
    if content:
        parts.append("")
        parts.append(content)

    text = "\n".join(parts).strip()

    meta = {
        "origin": item.get("source", "web"),
        "title": title,
        "url": url,
    }
    return text, meta


def _build_scoring_pairs(query: str, candidates: List[str]) -> List[Tuple[str, str]]:
    return [(query, c) for c in candidates]


def _batch_iter(iterable: Iterable[Any], batch_size: int):
    it = iter(iterable)
    while True:
        batch: List[Any] = []
        try:
            for _ in range(batch_size):
                batch.append(next(it))
        except StopIteration:
            if batch:
                yield batch
            break
        if not batch:
            break
        yield batch


# -----------------------------------------
# Async scoring via cross-encoder
# -----------------------------------------
async def _score_with_cross_encoder(
    model_name: str,
    pairs: List[Tuple[str, str]],
    batch_size: int = RERANK_BATCH_SIZE,
) -> List[float]:
    """
    Run CrossEncoder.predict in a background thread.
    """

    def _sync() -> List[float]:
        model = CrossEncoder(model_name, device="cpu")
        scores: List[float] = []
        for batch in _batch_iter(pairs, batch_size):
            batch_scores = model.predict(batch, show_progress_bar=False)
            scores.extend(float(s) for s in batch_scores)
        return scores

    return await asyncio.to_thread(_sync)


# -----------------------------------------
# Main Node
# -----------------------------------------
async def hybrid_reranker_node(state: MathState) -> MathState:
    """
    Hybrid Reranker Node.

    Conditional merging policy:

      - If needs_web_fallback is False (local strong):
          → rerank local_results + web_results together (web is a bonus).

      - If needs_web_fallback is True (local weak or empty):
          → rerank web_results only.

    Inputs:
      - state["query"]: str
      - state["local_results"]: List[Document]
      - state["web_results"]: List[dict]
      - state["needs_web_fallback"]: bool

    Outputs:
      - state["reranked_results"]: List[Document] with metadata:
          - origin: "local_rag" | "tavily" | "wikipedia" | "web_fetch" | "web"
          - rerank_score: float
          - url/title where applicable
    """
    query = (state.get("query") or "").strip()
    if not query:
        logger.warning("[RERANKER] Empty query → nothing to rerank.")
        state["reranked_results"] = []
        return state

    needs_web = bool(state.get("needs_web_fallback", False))
    local_docs: List[Document] = state.get("local_results") or []
    web_items: List[Dict[str, Any]] = state.get("web_results") or []

    candidates_text: List[str] = []
    candidates_meta: List[Dict[str, Any]] = []

    # -------------------------------------
    # Conditional merging logic
    # -------------------------------------
    if not needs_web:
        # Local is strong → include all local docs, plus any web results as bonus.
        logger.info("[RERANKER] Local RAG strong → reranking local + web.")
        for d in local_docs:
            text, meta = _make_candidate_from_local(d)
            if text:
                candidates_text.append(text)
                candidates_meta.append(meta)
    else:
        # Local is weak → rely primarily on web results.
        logger.info("[RERANKER] Local RAG weak → reranking web results only.")

    # Web candidates are included in both cases (if present)
    for w in web_items:
        text, meta = _make_candidate_from_web(w)
        if text:
            candidates_text.append(text)
            candidates_meta.append(meta)

    if not candidates_text:
        logger.warning("[RERANKER] No candidates from local or web → empty reranked_results.")
        state["reranked_results"] = []
        return state

    pairs = _build_scoring_pairs(query, candidates_text)

    try:
        scores = await _score_with_cross_encoder(RERANKER_MODEL, pairs)
    except Exception as exc:
        logger.exception("[RERANKER] Cross-encoder failed: %s", exc)
        # Fallback policy:
        if not needs_web:
            # Local was strong; fall back to local docs only.
            logger.info("[RERANKER] Falling back to local_results only.")
            state["reranked_results"] = list(local_docs[:RERANK_TOP_K])
        else:
            # Local was weak; fall back to raw web_items as simple Documents.
            logger.info("[RERANKER] Falling back to raw web_results only.")
            fallback_docs: List[Document] = []
            for w in web_items[:RERANK_TOP_K]:
                text, meta = _make_candidate_from_web(w)
                if not text:
                    continue
                fallback_docs.append(Document(page_content=text, metadata=meta))
            state["reranked_results"] = fallback_docs
        return state

    if len(scores) != len(candidates_text):
        logger.error(
            "[RERANKER] Score count (%d) != candidate count (%d).",
            len(scores),
            len(candidates_text),
        )

    scored: List[Tuple[str, Dict[str, Any], float]] = list(
        zip(candidates_text, candidates_meta, scores)
    )
    scored.sort(key=lambda x: x[2], reverse=True)

    final_docs: List[Document] = []
    for text, meta, score in scored[:RERANK_TOP_K]:
        m = dict(meta)
        m["rerank_score"] = float(score)
        final_docs.append(Document(page_content=text, metadata=m))

    state["reranked_results"] = final_docs
    logger.info("[RERANKER] Produced %d reranked results.", len(final_docs))
    return state
