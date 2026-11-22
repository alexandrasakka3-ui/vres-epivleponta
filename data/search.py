#!/usr/bin/env python3
"""Semantic search over the unified thesis table."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DATA_DIR = Path(__file__).resolve().parent
PARQUET_PATH = DATA_DIR / "unified.parquet"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
MODEL_NAME = "intfloat/multilingual-e5-small"
INSTITUTIONS = ("Θράκη", "Κρήτη", "Μακεδονία")
MAX_SEQ_LENGTH = 256

TEST_QUERIES = (
    "ανανεώσιμες πηγές ενέργειας",
    "μηχανική μάθηση στην ιατρική",
    "εκπαίδευση παιδιών με ειδικές ανάγκες",
    "τουρισμός και βιώσιμη ανάπτυξη",
    "αρχαία ελληνική γραμματεία",
)


@lru_cache(maxsize=1)
def _load_resources() -> tuple[SentenceTransformer, pd.DataFrame, np.ndarray]:
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LENGTH

    frame = pd.read_parquet(PARQUET_PATH)
    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    if embeddings.ndim != 2 or len(frame) != embeddings.shape[0]:
        raise ValueError(
            f"{EMBEDDINGS_PATH} has shape {embeddings.shape}, but "
            f"{PARQUET_PATH} has {len(frame)} rows."
        )

    return model, frame, embeddings.astype(np.float32)


def search(
    query: str,
    institutions: list[str] | None = None,
    top_k: int = 10,
) -> pd.DataFrame:
    """Return the highest-scoring rows for a query, with a ``score`` column."""
    if top_k < 0:
        raise ValueError("top_k must be non-negative.")
    if institutions is not None:
        invalid = set(institutions).difference(INSTITUTIONS)
        if invalid:
            raise ValueError(f"Unknown institutions: {sorted(invalid)}")

    model, frame, embeddings = _load_resources()
    query_embedding = model.encode(
        f"query: {query}",
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    scores = embeddings @ np.asarray(query_embedding, dtype=np.float32)

    if institutions is not None:
        eligible = frame["institution"].isin(institutions).to_numpy()
        scores = scores[eligible]
        result_frame = frame.loc[eligible]
    else:
        result_frame = frame

    result_count = min(top_k, len(result_frame))
    if result_count == 0:
        return result_frame.iloc[:0].assign(score=pd.Series(dtype=np.float32))

    if result_count < len(result_frame):
        candidate_indices = np.argpartition(scores, -result_count)[-result_count:]
        candidate_indices = candidate_indices[
            np.argsort(scores[candidate_indices])[::-1]
        ]
    else:
        candidate_indices = np.argsort(scores)[::-1]

    results = result_frame.iloc[candidate_indices].copy()
    results["score"] = scores[candidate_indices]
    return results


def main() -> None:
    output_lines = []
    for query in TEST_QUERIES:
        output_lines.append(f"Ερώτημα: {query}")
        for row in search(query, top_k=10).itertuples(index=False):
            title = str(row.title)[:80]
            output_lines.append(
                f"{row.institution} | {row.supervisor} | {title} | {row.score:.4f}"
            )
        output_lines.append("")

    output = "\n".join(output_lines).rstrip() + "\n"
    print(output, end="")
    (DATA_DIR / "search_test.txt").write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
