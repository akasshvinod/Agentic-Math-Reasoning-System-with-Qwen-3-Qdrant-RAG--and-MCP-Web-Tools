# src/ingestion/ingest_math.py

from __future__ import annotations

import json
import logging
import os
import uuid
from hashlib import md5
from pathlib import Path
from typing import Dict, Any, List, Iterable, Optional

from src.embedder.embedder import get_embedder
from src.tools.qdrant_tool import get_qdrant
from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
BATCH_SIZE = 64  # Slightly larger for throughput


# -------------------------------------------------------------
# JSONL Streaming
# -------------------------------------------------------------
def stream_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield each JSON object from a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("[INGEST] Skipping corrupt line: %s", line[:80])


def iter_all_records(max_records: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    """
    Stream all records from all JSONL files inside data/raw.
    Each file corresponds to a subject/split.

    max_records: optional cap for debugging/dry runs.
    """
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"[INGEST] RAW_DIR not found: {RAW_DIR}")

    count = 0
    for file_name in os.listdir(RAW_DIR):
        if not file_name.endswith(".jsonl"):
            continue

        file_path = RAW_DIR / file_name
        logger.info("[INGEST] Loading file: %s", file_path)

        for record in stream_jsonl(file_path):
            yield record
            count += 1
            if max_records is not None and count >= max_records:
                logger.info(
                    "[INGEST] Reached max_records=%d, stopping early for debug.",
                    max_records,
                )
                return


# -------------------------------------------------------------
# Build payload (metadata)
# -------------------------------------------------------------
def build_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare metadata payload for Qdrant."""
    return {
        "problem": record.get("problem", ""),
        "solution": record.get("solution", ""),
        "topic": record.get("topic", ""),
        "difficulty": record.get("difficulty", ""),
        "subject": record.get("subject", ""),
        "source_split": record.get("source_split", ""),
    }


def build_embedding_text(record: Dict[str, Any]) -> str:
    """
    Text to embed.

    For now: problem only (shorter index, faster), but you can switch
    to problem + short solution if needed.
    """
    problem = (record.get("problem") or "").strip()
    # If you want richer context, uncomment the next line:
    # solution = (record.get("solution") or "").strip()
    # return f"{problem}\nSolution: {solution}" if solution else problem
    return problem


def make_point_id(record: Dict[str, Any], fallback_counter: int) -> int:
    """
    Generate a stable numeric ID from record content.
    Uses MD5 hash truncated to 63 bits; falls back to counter if needed. [web:166]
    """
    problem = (record.get("problem") or "").strip()
    solution = (record.get("solution") or "").strip()
    raw = f"{problem}|||{solution}"
    if not raw.strip():
        # Completely empty record; fall back to a counter-based ID.
        return fallback_counter

    digest = md5(raw.encode("utf-8")).hexdigest()
    # Take lower 15 hex chars (~60 bits) to fit safely into signed 64-bit int.
    as_int = int(digest[-15:], 16)
    return as_int


# -------------------------------------------------------------
# Main Ingestion
# -------------------------------------------------------------
def ingest_dataset(
    batch_size: int = BATCH_SIZE,
    max_records: Optional[int] = None,
) -> None:
    """
    Main ingestion pipeline:
    - Streams all JSONL files
    - Embeds each problem
    - Upserts to Qdrant in batches

    max_records: if set, limits total records for faster debug runs.
    """
    logger.info("[INGEST] Starting ingestion from %s", RAW_DIR)

    qdrant = get_qdrant()
    embedder = get_embedder()

    vectors_batch: List[List[float]] = []
    payloads_batch: List[Dict[str, Any]] = []
    ids_batch: List[int] = []

    fallback_counter = 1
    record_count = 0
    upsert_count = 0

    for rec in iter_all_records(max_records=max_records):
        record_count += 1

        text = build_embedding_text(rec)
        if not text:
            continue  # skip empty problems

        payload = build_payload(rec)

        # Embed only the problem text (or problem+solution, per build_embedding_text)
        try:
            vec = embedder.embed(text, as_query=False)  # indexing, not query
        except Exception as exc:
            logger.warning("[INGEST] Failed to embed record %d: %s", record_count, exc)
            continue

        point_id = make_point_id(rec, fallback_counter)
        fallback_counter += 1

        vectors_batch.append(vec)
        payloads_batch.append(payload)
        ids_batch.append(point_id)

        # Batch insert
        if len(vectors_batch) >= batch_size:
            qdrant.upsert(ids_batch, vectors_batch, payloads_batch, wait=False)
            upsert_count += len(vectors_batch)

            logger.info("[INGEST] Upserted batch → total inserted: %d", upsert_count)

            vectors_batch.clear()
            payloads_batch.clear()
            ids_batch.clear()

        # Progress log
        if record_count % 1000 == 0:
            logger.info("[INGEST] Processed %d records...", record_count)

    # Final flush
    if vectors_batch:
        qdrant.upsert(ids_batch, vectors_batch, payloads_batch, wait=False)
        upsert_count += len(vectors_batch)
        logger.info("[INGEST] Final batch → total inserted: %d", upsert_count)

    logger.info(
        "[INGEST] COMPLETED. Total processed: %d, total inserted: %d",
        record_count,
        upsert_count,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Start with a small debug run to verify everything:
    # ingest_dataset(max_records=200)
    ingest_dataset()
