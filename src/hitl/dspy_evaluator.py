from __future__ import annotations

import logging
from typing import Dict, Any, Optional

import dspy  # still imported for future use if you add real evaluators

logger = logging.getLogger(__name__)


def _simple_coherence_score(text: str) -> float:
    """
    Very lightweight, local heuristic:
    - Penalize extremely short answers.
    - Reward presence of basic connective words.

    Returns a float in [0, 1].
    """
    t = (text or "").strip()
    if not t:
        return 0.0

    n_chars = len(t)
    if n_chars < 40:
        base = 0.2
    elif n_chars < 120:
        base = 0.5
    else:
        base = 0.8

    connectors = ["because", "therefore", "so ", "thus", "then", "hence"]
    has_conn = any(c in t.lower() for c in connectors)

    return min(1.0, base + (0.1 if has_conn else 0.0))


class MathQualityEvaluator:
    """
    Lightweight DSPy-based evaluator.

    Signals:
      - length_ok: answer not trivially short
      - has_final_answer: respects "Final answer:" contract
      - coherence: simple local heuristic in [0, 1]
      - human_feedback_positive / negative: simple sentiment flags
    """

    def evaluate(
        self,
        user_query: str,
        output_text: str,
        retrieved_docs: Optional[str] = None,
        human_feedback: Optional[str] = None,
    ) -> Dict[str, Any]:
        scores: Dict[str, Any] = {}

        text = (output_text or "").strip()
        fb = (human_feedback or "").strip().lower()

        # Basic heuristics
        scores["length_ok"] = len(text) > 40
        scores["has_final_answer"] = "final answer:" in text.lower()

        # Human feedback flags
        if fb:
            scores["human_feedback_positive"] = any(
                kw in fb for kw in ["good", "correct", "nice", "helpful", "clear"]
            )
            scores["human_feedback_negative"] = any(
                kw in fb for kw in ["wrong", "incorrect", "bad", "confusing", "unclear"]
            )
        else:
            scores["human_feedback_positive"] = None
            scores["human_feedback_negative"] = None

        # Coherence: use local heuristic instead of missing dspy.evaluate.coherence
        try:
            scores["coherence"] = _simple_coherence_score(text)
        except Exception as exc:
            logger.exception("[HITL] Local coherence heuristic failed: %s", exc)
            scores["coherence"] = None

        scores["retrieved_chars"] = len(retrieved_docs or "")

        return scores


_evaluator_instance: Optional[MathQualityEvaluator] = None


def get_dspy_evaluator() -> MathQualityEvaluator:
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = MathQualityEvaluator()
    return _evaluator_instance
