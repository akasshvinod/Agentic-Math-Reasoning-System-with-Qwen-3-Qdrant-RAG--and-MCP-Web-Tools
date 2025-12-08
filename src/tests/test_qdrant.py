# src/tests/test_qdrant.py

from __future__ import annotations

import math

from src.tools.qdrant_tool import get_qdrant
from src.embedder.embedder import get_embedder


def test_qdrant_basic_search():
    qdrant = get_qdrant()
    embedder = get_embedder()

    query = "Find the derivative of x^3"
    vec = embedder.embed(query)

    # Basic sanity checks on the query vector
    assert isinstance(vec, list)
    assert len(vec) == embedder.dim  # 1024 for bge-large-en-v1.5 [web:54]
    norm = math.sqrt(sum(v * v for v in vec))
    assert norm > 0 and math.isfinite(norm)

    results = qdrant.search(vec, top_k=3)

    print("\n[TEST] Basic Qdrant search results:")
    for r in results:
        print(r)

    # Results can be empty if you haven't ingested any data yet,
    # so only assert type/shape, not non-emptiness.
    assert isinstance(results, list)
    for item in results:
        assert "id" in item
        assert "score" in item
        assert "payload" in item


def test_qdrant_with_metadata_filter():
    """
    This will only return results once you store points
    with payloads like {"topic": "calculus"}.
    """
    qdrant = get_qdrant()
    embedder = get_embedder()

    query = "Find the derivative of x^3"
    vec = embedder.embed(query)

    filter_metadata = {"topic": "calculus"}
    results = qdrant.search(vec, top_k=3, filter_metadata=filter_metadata)

    print("\n[TEST] Qdrant search with metadata filter (topic=calculus):")
    for r in results:
        print(r)

    # Type/shape checks only; content depends on your ingested payloads.
    assert isinstance(results, list)
    for item in results:
        assert "payload" in item
        # If any results exist, they should respect the filter
        if item["payload"] is not None and "topic" in item["payload"]:
            assert item["payload"]["topic"] == "calculus"


if __name__ == "__main__":
    test_qdrant_basic_search()
    test_qdrant_with_metadata_filter()
