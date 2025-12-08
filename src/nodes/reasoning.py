from __future__ import annotations

import logging
from typing import Dict, Any, List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate  # [web:450]
from langchain_groq import ChatGroq  # Groq-native wrapper [web:482]

from src.config import QWEN_API_KEY, QWEN_MODEL
from src.graph.state import MathState

logger = logging.getLogger(__name__)


def _build_context_from_docs(docs: List[Document]) -> str:
    """
    Turn reranked Documents into a readable context block.

    Keeps math text from local RAG / web, but the LLM is instructed
    to answer in clean plain text (no LaTeX syntax).
    """
    if not docs:
        return "No external knowledge retrieved."

    chunks: List[str] = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        origin = meta.get("origin", "unknown")
        title = (meta.get("title") or "").strip()
        url = (meta.get("url") or "").strip()
        score = meta.get("rerank_score")

        header = f"[Source #{i} | origin={origin}"
        if isinstance(score, (int, float)):
            header += f" | score={score:.4f}"
        header += "]"

        if title:
            header += f"\nTitle: {title}"
        if url:
            header += f"\nURL: {url}"

        body = (doc.page_content or "").strip()[:2000]
        chunk = f"{header}\n\n{body}"
        chunks.append(chunk)

    return "\n\n\n".join(chunks)


async def reasoning_node(state: MathState) -> MathState:
    """
    Math Professor Reasoning Node (plain-text, with citations).

    Inputs:
      - state["query"]: str
      - state["reranked_results"]: List[Document]

    Output:
      - state["reasoning_output"]: str
    """
    query = (state.get("query") or "").strip()
    docs: List[Document] = state.get("reranked_results") or []

    logger.info("[REASONING] Generating plain-text reasoning for query: %r", query)

    retrieved_context = _build_context_from_docs(docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an expert mathematics tutor.\n"
                    "You receive a math problem and some retrieved sources "
                    "(from a math corpus and the web).\n"
                    "Your job is to give a clear, correct, human-friendly answer.\n"
                    "\n"
                    "STYLE RULES:\n"
                    "1. Explain your reasoning step by step in plain English.\n"
                    "2. Write all math in simple text, for example: x^2 = 16, x = 4 or x = -4.\n"
                    "3. Do NOT use LaTeX commands like \\frac, \\boxed, \\sqrt, $, or curly braces.\n"
                    "4. Avoid unnecessary meta-commentary; sound like a good teacher, not a system log.\n"
                    "5. If any source contradicts basic math, ignore that source.\n"
                    "6. If there is not enough information to solve the problem, say 'Insufficient information to determine the answer.'\n"
                    "7. When you use information from a retrieved source, reference it with a short tag like (Ref #1) matching [Source #1].\n"
                    "8. Only solve the math problem asked in the user question. Ignore unrelated parts of the retrieved documents."
                ),
            ),
            (
                "user",
                (
                    "User question:\n{query}\n\n"
                    "Retrieved knowledge (may or may not be needed):\n{context}\n\n"
                    "Now give a clear, step-by-step solution in plain text.\n"
                    "If a particular step is justified using retrieved knowledge, "
                    "include a reference tag like (Ref #2) that matches the source number.\n"
                    "Finish with a final line exactly in this format:\n"
                    "Final answer: ANSWER_HERE"
                ),
            ),
        ]
    )

    messages = prompt.format_messages(query=query, context=retrieved_context)

    llm = ChatGroq(
        model=QWEN_MODEL,      # e.g. "qwen/qwen3-32b"
        temperature=0.2,
        api_key=QWEN_API_KEY,
        max_retries=2,
    )

    try:
        raw_response = await llm.ainvoke(messages)
        response = raw_response.content if hasattr(raw_response, "content") else raw_response
        reasoning_output = response or ""
    except Exception as exc:
        logger.exception("[REASONING] LLM reasoning failed: %s", exc)
        reasoning_output = (
            "Error: The reasoning model failed. "
            "Please try again or verify the retrieved context."
        )

    state["reasoning_output"] = reasoning_output
    return state
