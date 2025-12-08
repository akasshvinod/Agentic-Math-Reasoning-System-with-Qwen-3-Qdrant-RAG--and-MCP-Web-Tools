from __future__ import annotations

import json
import logging
from typing import Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import MathState

from src.graph.state import MathState
from src.nodes.input_guardrail import input_guardrail_node
from src.nodes.router import router_node
from src.nodes.query_breaker import query_breaker_node
from src.nodes.local_rag import local_rag_node
from src.nodes.mcp_search import mcp_search_node
from src.nodes.hybrid_reranker import hybrid_reranker_node
from src.nodes.reasoning import reasoning_node
from src.nodes.verifier import verifier_node
from src.nodes.output_node import output_node
from src.nodes.feedback_node import feedback_node



logger = logging.getLogger(__name__)


def build_math_agent_graph() -> Any:
    """
    Build the full LangGraph pipeline for the Math Agent.

    Flow:
        START → input_guardrail → router → query_breaker → local_rag
        → (mcp_search?) → hybrid_reranker → reasoning → verifier
        → (mcp_search loop?) → output_guardrail → feedback → END
    """

    graph = StateGraph(MathState)

    # --------------------------
    # Register nodes
    # --------------------------
    graph.add_node("input_guardrail", input_guardrail_node)
    graph.add_node("router", router_node)
    graph.add_node("query_breaker", query_breaker_node)
    graph.add_node("local_rag", local_rag_node)
    graph.add_node("mcp_search", mcp_search_node)
    graph.add_node("reranker", hybrid_reranker_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("output_guardrail", output_node)
    graph.add_node("feedback", feedback_node)

    # --------------------------
    # START → Guardrail
    # --------------------------
    graph.add_edge(START, "input_guardrail")

    # --------------------------
    # Guardrail → Router or END
    # --------------------------
    def safe_or_end(state: MathState):
        return "router" if state.get("is_safe", False) else END

    graph.add_conditional_edges(
        "input_guardrail",
        safe_or_end,
        {
            "router": "router",
            END: END,
        },
    )

    # --------------------------
    # Router → QueryBreaker or END
    # --------------------------
    def route_math_or_end(state: MathState):
        is_math = state.get("is_math", True)
        return "query_breaker" if is_math else END

    graph.add_conditional_edges(
        "router",
        route_math_or_end,
        {
            "query_breaker": "query_breaker",
            END: END,
        },
    )

    # --------------------------
    # QueryBreaker → Local RAG
    # --------------------------
    graph.add_edge("query_breaker", "local_rag")

    # --------------------------
    # Local RAG → mcp_search (if weak) OR reranker (if strong)
    # --------------------------
    def rag_or_mcp(state: MathState):
        return "mcp_search" if state.get("needs_web_fallback", False) else "reranker"

    graph.add_conditional_edges(
        "local_rag",
        rag_or_mcp,
        {
            "mcp_search": "mcp_search",
            "reranker": "reranker",
        },
    )

    # MCP search → reranker
    graph.add_edge("mcp_search", "reranker")

    # Reranker → reasoning → verifier
    graph.add_edge("reranker", "reasoning")
    graph.add_edge("reasoning", "verifier")

    # --------------------------
    # Verifier → Loop (mcp_search) or Output
    # --------------------------
    MAX_VERIFICATION_LOOPS = 2

    def verification_loop(state: MathState):
        raw = state.get("verification")

        is_correct = True
        try:
            if isinstance(raw, str):
                data = json.loads(raw)
            elif isinstance(raw, dict):
                data = raw
            else:
                data = {}
            is_correct = bool(data.get("is_correct", True))
        except Exception:
            is_correct = True

        loop_count = state.get("loop_count", 0)

        # stop looping if we hit the cap
        if loop_count >= MAX_VERIFICATION_LOOPS:
            return "output_guardrail"

        # incorrect -> try another retrieval round
        return "output_guardrail" if is_correct else "mcp_search"

    graph.add_conditional_edges(
        "verifier",
        verification_loop,
        {
            "mcp_search": "mcp_search",
            "output_guardrail": "output_guardrail",
        },
    )

    # Output guardrail → feedback → END
    graph.add_edge("output_guardrail", "feedback")
    graph.add_edge("feedback", END)

    # --------------------------
    # Compile with MemorySaver
    # --------------------------
    checkpointer = MemorySaver()

    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=[],
    )

    logger.info("[GRAPH] Math Agent LangGraph built successfully.")
    return compiled


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    g = build_math_agent_graph()
    print("Graph compiled successfully.")
