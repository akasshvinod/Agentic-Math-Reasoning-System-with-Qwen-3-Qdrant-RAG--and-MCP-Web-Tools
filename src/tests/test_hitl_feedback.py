import asyncio

from src.nodes.local_rag import local_rag_node
from src.nodes.hybrid_reranker import hybrid_reranker_node
from src.nodes.reasoning import reasoning_node
from src.nodes.verifier import verifier_node
from src.nodes.output_node import output_node
from src.nodes.feedback_node import feedback_node


async def run_case(q: str, feedback: str):
    state = {"query": q}

    # 1) Retrieval + reranking
    state = await local_rag_node(state)
    state = await hybrid_reranker_node(state)

    # 2) Reasoning
    state = await reasoning_node(state)

    # 3) Verification
    state = await verifier_node(state)

    # 4) Output formatting
    state = await output_node(state)
    print("\n" + "=" * 80)
    print("QUERY:", q)
    print("\n[FINAL OUTPUT]\n")
    print(state.get("final_output", ""))

    # 5) Simulated human feedback
    state["feedback"] = feedback

    # 6) HITL logging (DSPy + JSONL)
    state = await feedback_node(state)
    print("\n[HITL] Feedback logged.\n")


async def main():
    await run_case(
        "Find x such that x^2 = 16.",
        feedback="Good answer, but could be a bit shorter."
    )


if __name__ == "__main__":
    asyncio.run(main())
