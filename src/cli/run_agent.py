from __future__ import annotations

import asyncio
import logging
import textwrap
import uuid
from typing import Optional

from src.graph.build_graph import build_math_agent_graph
from src.graph.state import MathState


logger = logging.getLogger(__name__)


BANNER = r"""
┌───────────────────────────────────────────────┐
│           Math Agent - CLI Interface         │
└───────────────────────────────────────────────┘
Type your math question (calculus, algebra,
limits, series, etc.).

Commands:
  /exit, /quit   → end session
  /clear         → clear screen
"""


def _wrap(
    text: str,
    width: int = 88,
    indent: str = "",
    sub_indent: str = "",
) -> str:
    """Utility: wrap text for console display."""
    return textwrap.fill(
        text.strip(),
        width=width,
        initial_indent=indent,
        subsequent_indent=sub_indent or indent,
    )


def format_answer_structured(answer: str) -> str:
    """
    Format final answer as:
      - Summary line (first sentence)
      - Stepwise bullets (remaining sentences)
    """
    answer = (answer or "").strip()
    if not answer:
        return ""

    sentences = [s.strip() for s in answer.replace("\n", " ").split(".") if s.strip()]
    if not sentences:
        return ""

    summary = sentences[0]
    steps = sentences[1:]

    lines = []
    lines.append("\n╭─ ANSWER ────────────────────────────────────────────────╮")
    lines.append("│ Summary:")
    for line in _wrap(summary, width=70).splitlines():
        lines.append(f"│   {line}")
    if steps:
        lines.append("│")
        lines.append("│ Steps:")
        for s in steps:
            wrapped = _wrap(s, width=70, indent="  • ", sub_indent="    ")
            for line in wrapped.splitlines():
                lines.append(f"│ {line}")
    lines.append("╰──────────────────────────────────────────────────────────╯\n")
    return "\n".join(lines)


def format_verification_block(verification: str) -> str:
    """Pretty-print the verifier JSON/text as a block."""
    verification = (verification or "").strip()
    if not verification:
        return ""
    lines = []
    lines.append("\n╭─ VERIFICATION ─────────────────────────────────────────╮")
    for line in _wrap(verification, width=72).splitlines():
        lines.append(f"│ {line}")
    lines.append("╰────────────────────────────────────────────────────────╯\n")
    return "\n".join(lines)


async def run_math_agent_cli(thread_id: Optional[str] = None) -> None:
    """
    Run an interactive CLI loop for the Math Agent.

    - Builds the LangGraph math agent once.
    - Uses a stable thread_id so MemorySaver can persist state.
    - For each user query:
        - Runs the full graph pipeline.
        - Prints a structured final answer and verifier summary.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print(BANNER)

    graph = build_math_agent_graph()

    if thread_id is None:
        thread_id = f"cli-session-{uuid.uuid4().hex[:8]}"
    print(f"[session id: {thread_id}]\n")

    while True:
        try:
            user_q = input("You  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.\n")
            break

        if not user_q:
            continue

        # Commands
        cmd = user_q.lower()
        if cmd in {"/exit", "exit", "/quit", "quit"}:
            print("Session ended.\n")
            break
        if cmd in {"/clear", "clear"}:
            print("\033c", end="")
            print(BANNER)
            print(f"[session id: {thread_id}]\n")
            continue

        state: MathState = {"query": user_q}

        print("\nAgent> Thinking...\n")

        try:
            result: MathState = await graph.ainvoke(
                state,
                config={"configurable": {"thread_id": thread_id}},
            )
        except Exception as exc:
            logger.exception("Graph invocation failed: %s", exc)
            print("Agent> Sorry, something went wrong while solving this problem.\n")
            continue

        final_answer = (result.get("final_output") or "").strip()
        verification = (result.get("verification") or "").strip()
        loop_count = int(result.get("loop_count", 0))

        # Structured final answer
        if final_answer:
            print(format_answer_structured(final_answer))
        else:
            print("Agent> I could not produce an answer for this query.\n")

        # Verifier block (compact JSON/string)
        if verification:
            print(format_verification_block(verification))

        if loop_count > 0:
            print(f"[info] verifier loop iterations: {loop_count}\n")

        try:
            fb = input("Feedback (enter to skip) > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.\n")
            break

        if fb:
            # At the moment this is just acknowledged in CLI;
            # your feedback_node logs feedback when present in state.
            print("[noted] Feedback captured in CLI (not yet wired into graph).\n")
        else:
            print()


def main() -> None:
    asyncio.run(run_math_agent_cli())


if __name__ == "__main__":
    main()
