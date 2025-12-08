import asyncio
from pprint import pprint

from src.nodes.mcp_search import mcp_search_node


async def run_case(q: str):
    state = {"query": q}

    try:
        out = await asyncio.wait_for(mcp_search_node(state), timeout=20)
    except asyncio.TimeoutError:
        print("\n" + "=" * 80)
        print("QUERY:", q)
        print("[ERROR] mcp_search_node timed out after 20 seconds.")
        return

    print("\n" + "=" * 80)
    print("QUERY:", q)
    print("num web_results:", len(out.get("web_results", [])))

    if out.get("web_results"):
        first = out["web_results"][0]
        print("top source:", first.get("source"))
        print("top title:", first.get("title"))
        print("top url:", first.get("url"))
        content = (first.get("content") or "")[:400]
        print("content preview:", content.replace("\n", " ")[:400])


async def main():
    await run_case("linear algebra basics")
    await run_case("what is the fundamental theorem of calculus?")
    await run_case("RAG for mathematical problem solving research 2024")
    await run_case("history of the Pythagorean theorem")


if __name__ == "__main__":
    asyncio.run(main())
