# src/tests/test_reasoning_node.py

import asyncio

from src.nodes.local_rag import local_rag_node
from src.nodes.hybrid_reranker import hybrid_reranker_node
from src.nodes.reasoning import reasoning_node


async def run_case(q: str):
    state = {"query": q}

    # 1) Local RAG
    state = await local_rag_node(state)
    print("\n" + "=" * 80)
    print("QUERY:", q)
    print("local_results:", len(state.get("local_results", [])))
    print("needs_web_fallback:", state.get("needs_web_fallback"))

    # 2) (Optional) web fallback is off here to test pure local behavior
    # If you want full pipeline incl. web, also call mcp_search_node before reranker.

    # 3) Hybrid reranker (local only if needs_web_fallback=False)
    state = await hybrid_reranker_node(state)
    print("reranked_results:", len(state.get("reranked_results", [])))

    # 4) Reasoning node
    state = await reasoning_node(state)

    print("\n[REASONING OUTPUT]\n")
    print(state.get("reasoning_output", ""))


async def main():
    # Test with a query that clearly hits Hendrycks MATH
    await run_case("Find x if x^2 = 16")


if __name__ == "__main__":
    asyncio.run(main())
