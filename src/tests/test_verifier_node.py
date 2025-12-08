import asyncio

from src.nodes.local_rag import local_rag_node
from src.nodes.hybrid_reranker import hybrid_reranker_node
from src.nodes.reasoning import reasoning_node
from src.nodes.verifier import verifier_node


async def run_case(q: str):
    state = {"query": q}

    # 1) Local RAG (and web fallback if your graph does that here)
    state = await local_rag_node(state)

    # 2) Rerank retrieved docs
    state = await hybrid_reranker_node(state)

    # 3) Generate step‑by‑step solution
    state = await reasoning_node(state)
    print("\n" + "=" * 80)
    print("QUERY:", q)
    print("\n[REASONING OUTPUT]\n")
    print(state.get("reasoning_output", ""))

    # 4) Verify solution
    state = await verifier_node(state)
    print("\n[VERIFICATION JSON]\n")
    print(state.get("verification", ""))


async def main():
    await run_case("Find x such that x^2 = 16.")


if __name__ == "__main__":
    asyncio.run(main())
