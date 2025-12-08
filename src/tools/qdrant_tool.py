# src/tools/qdrant_tool.py

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
from src.embedder.embedder import get_embedder

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """
    Production-ready Qdrant vector store wrapper.

    Responsibilities:
    - Manage Qdrant client connection
    - Ensure collection exists (dimension, distance)
    - Upsert batches of points (id + vector + payload)
    - Perform vector search with optional metadata filters
    """

    def __init__(self, collection_name: str | None = None) -> None:
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY or None,
            timeout=30,
        )
        self.collection = collection_name or QDRANT_COLLECTION

        logger.info("[QDRANT] Connected to %s, collection=%s", QDRANT_URL, self.collection)
        self._ensure_collection()

    # -------------------------------------------------------------
    # Collection management
    # -------------------------------------------------------------
    def _ensure_collection(self) -> None:
        embedder = get_embedder()
        vector_dim = embedder.dim

        try:
            collections = self.client.get_collections().collections
            existing_names = {c.name for c in collections}
        except Exception as exc:
            logger.exception("[QDRANT] Failed to list collections: %s", exc)
            raise RuntimeError("Unable to connect to Qdrant") from exc

        if self.collection in existing_names:
            logger.info(
                "[QDRANT] Collection '%s' exists (expected dim=%d, distance=COSINE).",
                self.collection,
                vector_dim,
            )
            # Optional: fetch collection info and assert vector size/distance.
            return

        logger.warning(
            "[QDRANT] Collection '%s' missing → creating with dim=%d, COSINE.",
            self.collection,
            vector_dim,
        )

        try:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=vector_dim,
                    distance=models.Distance.COSINE,
                ),
            )
        except UnexpectedResponse as exc:
            logger.exception("[QDRANT] Failed to create collection: %s", exc)
            raise RuntimeError("Failed to create Qdrant collection") from exc

        logger.info("[QDRANT] Collection '%s' created successfully.", self.collection)

    # -------------------------------------------------------------
    # Upsert
    # -------------------------------------------------------------
    def upsert(
        self,
        ids: List[int],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        wait: bool = False,
    ) -> None:
        if not (len(ids) == len(vectors) == len(payloads)):
            raise ValueError("IDs, vectors, and payloads must have same length")

        if not ids:
            logger.warning("[QDRANT] upsert called with empty batch; skipping.")
            return

        points = [
            models.PointStruct(id=pid, vector=vec, payload=payload)
            for pid, vec, payload in zip(ids, vectors, payloads)
        ]

        logger.info("[QDRANT] Upserting %d points into '%s'...", len(points), self.collection)
        try:
            self.client.upsert(
                collection_name=self.collection,
                points=points,
                wait=wait,
            )
        except Exception as exc:
            logger.exception("[QDRANT] Upsert failed: %s", exc)
            raise RuntimeError("Qdrant upsert failed") from exc

    # -------------------------------------------------------------
    # Filter builder (metadata)
    # -------------------------------------------------------------
    def _build_filter(
        self,
        filter_metadata: Optional[Dict[str, Any]],
    ) -> Optional[models.Filter]:
        """
        Build a simple AND filter from a metadata dict.
        Example: {"topic": "calculus", "difficulty": "hard"}.
        """
        if not filter_metadata:
            return None

        conditions: List[models.FieldCondition] = []
        for key, value in filter_metadata.items():
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
            )

        # AND semantics over all conditions. [web:124]
        return models.Filter(must=conditions) if conditions else None

    # -------------------------------------------------------------
    # Search
    # -------------------------------------------------------------
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Vector search with optional metadata filter.

        Args:
            query_vector: Dense embedding of the query.
            top_k: Number of results to return.
            filter_metadata: Equality filters, e.g. {"topic": "calculus", "difficulty": "hard"}.
        """
        if not query_vector:
            raise ValueError("query_vector must be non-empty")

        q_filter = self._build_filter(filter_metadata)

        try:
            # Use query_points according to current qdrant-client API. [web:145][web:147]
            results = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=top_k,
                query_filter=q_filter,
                with_payload=True,
            )
        except Exception as exc:
            logger.exception("[QDRANT] Search failed: %s", exc)
            raise RuntimeError("Qdrant search failed") from exc

        formatted: List[Dict[str, Any]] = []
        for r in results.points:
            formatted.append(
                {
                    "id": r.id,
                    "score": r.score,
                    "payload": r.payload,
                }
            )

        return formatted


# Singleton ------------------------------------------------------

_qdrant_instance: Optional[QdrantVectorStore] = None


def get_qdrant() -> QdrantVectorStore:
    global _qdrant_instance
    if _qdrant_instance is None:
        _qdrant_instance = QdrantVectorStore()
    return _qdrant_instance
