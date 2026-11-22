#!/usr/bin/env python3
"""Build resumable, normalized multilingual-e5 embeddings for unified.parquet."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DATA_DIR = Path(__file__).resolve().parent
PARQUET_PATH = DATA_DIR / "unified.parquet"
CHUNKS_DIR = DATA_DIR / "chunks"
OUTPUT_PATH = DATA_DIR / "embeddings.npy"
MODEL_NAME = "intfloat/multilingual-e5-small"
CHUNK_SIZE = 2_000
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 256
MAX_TEXT_LENGTH = 1_500


def _text_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _make_passages(frame: pd.DataFrame) -> list[str]:
    return [
        f"passage: {_text_value(title)}. {_text_value(abstract)}"[:MAX_TEXT_LENGTH]
        for title, abstract in zip(frame["title"], frame["abstract"])
    ]


def _load_model() -> SentenceTransformer:
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def _encode(model: SentenceTransformer, passages: list[str]) -> np.ndarray:
    embeddings = model.encode(
        passages,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float16)


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as output_file:
        np.save(output_file, array)
    temporary_path.replace(path)


def _save_test_result(frame: pd.DataFrame, model: SentenceTransformer) -> None:
    test_frame = frame.head(200)
    passages = _make_passages(test_frame)
    started = time.perf_counter()
    embeddings = _encode(model, passages)
    elapsed = time.perf_counter() - started
    estimated_total = elapsed / len(passages) * len(frame)
    print(
        f"Test: {len(passages)} rows encoded in {elapsed:.2f} seconds "
        f"({elapsed / len(passages):.3f} sec/row).",
        flush=True,
    )
    print(
        f"Estimated encoding time for {len(frame):,} rows at this rate: "
        f"{estimated_total / 60:.1f} minutes (excluding model loading).",
        flush=True,
    )
    print(f"Test embedding shape: {embeddings.shape}; dtype: {embeddings.dtype}", flush=True)


def _build_all(frame: pd.DataFrame, model: SentenceTransformer) -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    chunk_count = math.ceil(len(frame) / CHUNK_SIZE)
    pending = [
        index
        for index in range(chunk_count)
        if not (CHUNKS_DIR / f"embeddings_{index:05d}.npy").exists()
    ]
    pending_rows = sum(
        min(CHUNK_SIZE, len(frame) - index * CHUNK_SIZE) for index in pending
    )
    print(
        f"Building {len(frame):,} rows in {chunk_count} chunks; "
        f"{len(pending)} chunks ({pending_rows:,} rows) remaining.",
        flush=True,
    )

    started = time.perf_counter()
    encoded_rows = 0
    for completed, index in enumerate(pending, start=1):
        first = index * CHUNK_SIZE
        last = min(first + CHUNK_SIZE, len(frame))
        chunk_path = CHUNKS_DIR / f"embeddings_{index:05d}.npy"
        embeddings = _encode(model, _make_passages(frame.iloc[first:last]))
        if embeddings.shape[0] != last - first:
            raise RuntimeError(
                f"Chunk {index} has {embeddings.shape[0]} embeddings; expected {last - first}."
            )
        _save_npy_atomic(chunk_path, embeddings)
        encoded_rows += last - first
        elapsed = time.perf_counter() - started
        rate = encoded_rows / elapsed if elapsed else 0.0
        remaining_rows = pending_rows - encoded_rows
        eta_seconds = remaining_rows / rate if rate else 0.0
        print(
            f"Chunk {index + 1}/{chunk_count} saved "
            f"({last:,}/{len(frame):,} rows); "
            f"{completed}/{len(pending)} pending chunks complete; "
            f"ETA {eta_seconds / 60:.1f} min.",
            flush=True,
        )

    chunk_arrays = []
    for index in range(chunk_count):
        chunk_path = CHUNKS_DIR / f"embeddings_{index:05d}.npy"
        if not chunk_path.exists():
            raise FileNotFoundError(f"Missing expected embedding chunk: {chunk_path}")
        chunk = np.load(chunk_path, mmap_mode="r")
        expected_rows = min(CHUNK_SIZE, len(frame) - index * CHUNK_SIZE)
        if chunk.shape[0] != expected_rows:
            raise ValueError(
                f"{chunk_path} has {chunk.shape[0]} rows; expected {expected_rows}."
            )
        chunk_arrays.append(chunk)

    combined = np.concatenate(chunk_arrays, axis=0)
    if combined.shape[0] != len(frame):
        raise RuntimeError(
            f"Combined embeddings have {combined.shape[0]} rows; expected {len(frame)}."
        )
    _save_npy_atomic(OUTPUT_PATH, combined)
    print(
        f"Saved {OUTPUT_PATH} with shape {combined.shape}, dtype {combined.dtype}; "
        f"file size {OUTPUT_PATH.stat().st_size / (1024 * 1024):.2f} MiB.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test",
        action="store_true",
        help="Encode only the first 200 rows and print timing without saving chunks.",
    )
    args = parser.parse_args()

    frame = pd.read_parquet(PARQUET_PATH)
    required_columns = {"title", "abstract"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"{PARQUET_PATH} is missing columns: {sorted(missing_columns)}")

    print(f"Loaded {len(frame):,} rows from {PARQUET_PATH}.", flush=True)
    model_started = time.perf_counter()
    model = _load_model()
    print(
        f"Loaded {MODEL_NAME} (max_seq_length={MAX_SEQ_LENGTH}) in "
        f"{time.perf_counter() - model_started:.2f} seconds.",
        flush=True,
    )

    if args.test:
        _save_test_result(frame, model)
    else:
        _build_all(frame, model)


if __name__ == "__main__":
    main()
