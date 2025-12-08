# src/embedder/embedder.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional

from sentence_transformers import SentenceTransformer
from src.config import EMBED_MODEL

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbedConfig:
    model_name: str = EMBED_MODEL  # e.g. "BAAI/bge-large-en-v1.5"
    batch_size: int = 32
    normalize: bool = True          # cosine / dot-product friendly
    use_query_instruction: bool = True


class Embedder:
    """
    Production-ready embedder using SentenceTransformers
    (BAAI/bge-large-en-v1.5, 1024-dim).
    """

    def __init__(self, config: Optional[EmbedConfig] = None):
        self.config = config or EmbedConfig()
        self.model_name = self.config.model_name

        logger.info("Loading embedder model: %s", self.model_name)
        self.model = SentenceTransformer(self.model_name)
        # bge-large-en-v1.5 → 1024-dim output
        self.dim: int = self.model.get_sentence_embedding_dimension()

        logger.info("Embedder loaded: %s (dim=%d)", self.model_name, self.dim)

    # --- internal helpers -------------------------------------------------

    def _prepare_query(self, text: str) -> str:
        """
        Apply query instruction for retrieval, if enabled.
        BGE docs recommend a short instruction for best performance.
        """
        if not self.config.use_query_instruction:
            return text

        # Keep this short; can be tweaked later from config.
        prefix = "Represent this sentence for retrieving relevant math problems: "
        return prefix + text

    def _validate_text(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("text is empty after stripping")
        return cleaned

    # --- public API -------------------------------------------------------

    def embed(self, text: str, as_query: bool = True) -> List[float]:
        """
        Embed a single string into a vector.

        Args:
            text: Input string.
            as_query: If True, apply query instruction prefix (for search).
        """
        cleaned = self._validate_text(text)
        to_encode = self._prepare_query(cleaned) if as_query else cleaned

        try:
            vec = self.model.encode(
                to_encode,
                batch_size=1,
                normalize_embeddings=self.config.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.exception("Failed to embed text: %s", exc)
            raise RuntimeError("Embedding failed") from exc

        return vec.tolist()

    def embed_batch(
        self,
        texts: Iterable[str],
        as_query: bool = True,
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Embed multiple texts with batching.

        Args:
            texts: Iterable of input strings.
            as_query: If True, apply query instruction to each item.
            batch_size: Override default batch size if provided.
        """
        texts = list(texts)
        if not texts:
            raise ValueError("embed_batch received an empty list")

        cleaned = [self._validate_text(t) for t in texts]
        if as_query:
            cleaned = [self._prepare_query(t) for t in cleaned]

        bs = batch_size or self.config.batch_size

        try:
            result = self.model.encode(
                cleaned,
                batch_size=bs,
                normalize_embeddings=self.config.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.exception("Failed to embed batch of size %d: %s", len(cleaned), exc)
            raise RuntimeError("Batch embedding failed") from exc

        return [v.tolist() for v in result]


# Singleton -----------------------------------------------------------------

_embedder_instance: Optional[Embedder] = None


def get_embedder() -> Embedder:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder()
    return _embedder_instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    embedder = get_embedder()
    text = "Solve x^2 - 5x + 6 = 0"
    vec = embedder.embed(text)

    print("Model:", embedder.model_name)
    print("Dim:", embedder.dim)
    print("First 10 dims:", vec[:10])
