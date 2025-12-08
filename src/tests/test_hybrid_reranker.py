# src/tests/test_hybrid_reranker.py

import asyncio
from pprint import pprint

from src.nodes.local_rag import local_rag_node
from src.nodes.mcp_search import mcp_search_node
from src.nodes.hybrid_reranker import hybrid_reranker_node


async def run_case(q: str, force_web: bool = False):
    state = {"query": q}

    # 1) Local RAG
    state = await local_rag_node(state)
    print("\n" + "=" * 80)
    print("QUERY:", q)
    print("local_results:", len(state.get("local_results", [])))
    print("needs_web_fallback after local_rag:", state.get("needs_web_fallback"))

    # 2) Optional web fallback (simulate both strong + weak cases)
    if force_web:
        state["needs_web_fallback"] = True

    if state.get("needs_web_fallback"):
        state = await mcp_search_node(state)
        print("web_results:", len(state.get("web_results", [])))
    else:
        print("Skipping MCP search (local strong).")

    # 3) Hybrid reranker
    state = await hybrid_reranker_node(state)

    reranked = state.get("reranked_results", [])
    print("reranked_results:", len(reranked))

    for i, doc in enumerate(reranked[:5]):
        print(f"\n  [#{i}] origin={doc.metadata.get('origin')}, "
              f"score={doc.metadata.get('rerank_score'):.4f}")
        print("      title:", doc.metadata.get("title"))
        print("      url:", doc.metadata.get("url"))
        print("      preview:", doc.page_content[:200].replace("\n", " "))


async def main():
    # Case 1: typical math query, expect strong local RAG, no fallback
    await run_case("Find x if x^2 = 16")

    # Case 2: more conceptual / web-heavy, likely to trigger web fallback
    await run_case("RAG for mathematical problem solving research 2024", force_web=True)


if __name__ == "__main__":
    asyncio.run(main())
