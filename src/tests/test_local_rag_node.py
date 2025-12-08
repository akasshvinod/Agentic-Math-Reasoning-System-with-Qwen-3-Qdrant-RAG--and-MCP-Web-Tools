# src/tests/test_local_rag_node.py

import asyncio
from pprint import pprint

from src.nodes.local_rag import local_rag_node


async def main():
    # Simple math query
    state_simple = {"query": "Find x if x^2 = 16"}
    out_simple = await local_rag_node(state_simple)

    print("\n[SIMPLE QUERY TEST]")
    print("needs_web_fallback:", out_simple.get("needs_web_fallback"))
    print("num local_results:", len(out_simple.get("local_results", [])))
    if out_simple.get("local_results"):
        first = out_simple["local_results"][0]
        print("top score:", first.metadata.get("score"))
        print("top doc preview:\n", first.page_content[:300])

    # Multi-step query with broken_queries populated (simulating query_breaker)
    state_broken = {
        "query": "First solve x^2 = 16 and then find derivative of x^3.",
        "broken_queries": ["First solve x^2 = 16", "then find derivative of x^3."],
    }
    out_broken = await local_rag_node(state_broken)

    print("\n[BROKEN QUERY TEST]")
    print("retrieval used:", state_broken["broken_queries"][0])
    print("needs_web_fallback:", out_broken.get("needs_web_fallback"))
    print("num local_results:", len(out_broken.get("local_results", [])))
    if out_broken.get("local_results"):
        first = out_broken["local_results"][0]
        print("top score:", first.metadata.get("score"))
        print("top doc preview:\n", first.page_content[:300])


if __name__ == "__main__":
    asyncio.run(main())
