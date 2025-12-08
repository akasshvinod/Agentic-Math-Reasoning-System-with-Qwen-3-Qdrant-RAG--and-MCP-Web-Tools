from __future__ import annotations

import json
import logging
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.config import GPT_API_KEY, GPT_MODEL, VERIFIER_TEMPERATURE

logger = logging.getLogger(__name__)


VERIFIER_SCHEMA = """
The output MUST be a strict JSON object exactly in this schema:

{
  "is_correct": true/false,
  "issues": [list of strings],
  "improved_answer": "string"
}

Rules:
- "is_correct": true only if the reasoning AND the final answer are mathematically correct.
- "issues": include any logical errors, missing steps, wrong calculations, or unsafe assumptions.
- "improved_answer": MUST be a corrected, clean, step-by-step answer using plain text math.
- Do NOT use LaTeX. Use forms like: x^2, 1/2, sqrt(x), x = 4 or x = -4.
- No markdown, no extra commentary, and no text before or after the JSON object.
"""


def _strip_code_fence(text: str) -> str:
    """
    Best-effort removal of `````` or `````` wrappers.
    Keeps inner JSON intact.
    """
    if not isinstance(text, str):
        return ""

    t = text.strip()

    # Handle fenced code blocks like `````` or ``````
    if t.startswith("``````"):
        t = t[3:-3].strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()

    return t


async def verifier_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verifier Node (LLM-as-a-judge).

    Uses a Groq-hosted GPT model (e.g., openai/gpt-oss-20b) to evaluate
    the reasoning_output produced by the reasoning node.

    Inputs in state:
      - "query": original math problem (str)
      - "reasoning_output": model's step-by-step solution (str)

    Output in state:
      - "verification": JSON string matching VERIFIER_SCHEMA
      - increments "loop_count" when answer is incorrect
    """

    query = (state.get("query") or "").strip()
    reasoning_output = (state.get("reasoning_output") or "").strip()

    logger.info("[VERIFIER] Evaluating reasoning for query: %r", query)

    verifier_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a strict but fair mathematical verifier.\n"
                    "You receive the original problem and a proposed solution.\n"
                    "Your job is to judge ONLY the math: correctness, clarity, and logic.\n"
                    "\n"
                    "Follow these instructions carefully:\n"
                    "1. Check whether every key step is mathematically valid.\n"
                    "2. Check whether the final numerical/symbolic answer is correct.\n"
                    "3. Note any errors, gaps, or confusing explanations.\n"
                    "4. If needed, rewrite the solution into a clean, correct, "
                    "step-by-step answer in plain text math.\n"
                    "5. Respond ONLY with a single JSON object, no extra text.\n"
                    f"{VERIFIER_SCHEMA.replace('{', '{{').replace('}', '}}')}"
                ),
            ),
            (
                "user",
                (
                    "Problem:\n{query}\n\n"
                    "Solution to verify:\n{solution}\n\n"
                    "Now evaluate this solution strictly according to the schema."
                ),
            ),
        ]
    )

    messages = verifier_prompt.format_messages(
        query=query,
        solution=reasoning_output,
    )

    llm = ChatGroq(
        model=GPT_MODEL,                 # e.g. "openai/gpt-oss-20b" via Groq
        temperature=VERIFIER_TEMPERATURE,  # 0.0 recommended
        api_key=GPT_API_KEY,
        max_retries=2,
    )

    try:
        raw_response = await llm.ainvoke(messages)
        content = raw_response.content if hasattr(raw_response, "content") else raw_response
        if not isinstance(content, str):
            raise ValueError("Verifier response is not a string.")

        content = _strip_code_fence(content)

        if not content.startswith("{"):
            raise ValueError(f"Verifier did not return bare JSON: {content[:80]}")

        # Validate JSON; if this fails, we go to fallback
        json.loads(content)

        verification = content

    except Exception as exc:
        logger.exception("[VERIFIER] Failed: %s", exc)
        verification = json.dumps(
            {
                "is_correct": False,
                "issues": ["Verifier model failed to evaluate the solution."],
                "improved_answer": (
                    "The system could not verify this solution. "
                    "Please try again or solve the problem manually."
                ),
            },
            ensure_ascii=False,
        )

    # Store raw verification (JSON string)
    state["verification"] = verification

    # --- Increment loop_count if answer is marked incorrect ---
    try:
        data = json.loads(verification) if isinstance(verification, str) else verification
    except Exception:
        data = {}

    if isinstance(data, dict) and not data.get("is_correct", True):
        state["loop_count"] = state.get("loop_count", 0) + 1

    return state
