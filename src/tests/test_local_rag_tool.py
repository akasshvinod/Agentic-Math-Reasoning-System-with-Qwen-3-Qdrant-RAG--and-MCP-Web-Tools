# src/tests/test_local_rag_tool.py

from __future__ import annotations

import pprint

from src.tools.local_rag_tool import local_rag_search


def test_local_rag_basic():
    out = local_rag_search.invoke(
        {
            "query": "Find x if x^2 = 16",
            "top_k": 3,
            # "topic": "Algebra",  # optional
        }
    )

    print("\n[TEST] Local RAG Tool Output:")
    pprint.pprint(out)

    assert isinstance(out, dict)
    assert "results" in out
    assert "used_filters" in out
    assert isinstance(out["used_filters"].get("fallback"), bool)

    if out["results"]:
        first = out["results"][0]
        assert "id" in first
        assert "score" in first
        assert "payload" in first


if __name__ == "__main__":
    test_local_rag_basic()
