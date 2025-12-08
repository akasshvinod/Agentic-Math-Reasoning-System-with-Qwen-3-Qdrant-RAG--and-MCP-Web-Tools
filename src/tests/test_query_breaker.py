# tests/test_query_breaker.py

import asyncio
from src.nodes.query_breaker import query_breaker_node


async def main():
    # Heuristic path: obvious multi-step query
    state_heuristic = {
        "query": "First solve x^2 = 16 and then find derivative of x^3."
    }
    out_heuristic = await query_breaker_node(state_heuristic)
    print("\n[HEURISTIC TEST]")
    print("Query:", state_heuristic["query"])
    print("broken_queries:", out_heuristic.get("broken_queries"))

    # LLM path: no obvious trigger words, so heuristics return single item
    state_llm = {
        "query": "Farmer Alfred has three times as many chickens as cows. In total, there are 60 legs in the barn. How many cows does Farmer Alfred have?"
    }
    out_llm = await query_breaker_node(state_llm)
    print("\n[LLM TEST]")
    print("Query:", state_llm["query"])
    print("broken_queries:", out_llm.get("broken_queries"))


if __name__ == "__main__":
    asyncio.run(main())
