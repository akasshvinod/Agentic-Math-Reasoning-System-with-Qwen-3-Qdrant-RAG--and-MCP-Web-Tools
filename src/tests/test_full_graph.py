# src/tests/test_full_graph.py

import asyncio
import time
import logging

from src.graph.build_graph import build_math_agent_graph
from src.graph.state import MathState


logging.basicConfig(
    level=logging.INFO,  # switch to DEBUG if you want full traces
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


async def run_case(query: str, feedback: str | None = None):
    graph = build_math_agent_graph()

    state: MathState = {"query": query}
    if feedback is not None:
        state["feedback"] = feedback

    print("\n" + "=" * 80)
    print("QUERY:", query)

    start = time.time()
    result = await graph.ainvoke(
        state,
        config={"configurable": {"thread_id": "test-thread-1"}},
    )
    print(f"\n[INFO] Graph finished in {time.time() - start:.2f}s\n")

    print("[STATE KEYS]", list(result.keys()))
    print("\n[FINAL OUTPUT]\n")
    print(result.get("final_output", ""))
    print("\n[VERIFICATION]\n")
    print(result.get("verification", ""))
    print("\n[LOOP COUNT]:", result.get("loop_count", 0))


async def main():
     await run_case(
        "Who is the prime minister of India",
        feedback="solve carefully"
    )

if __name__ == "__main__":
    asyncio.run(main())
