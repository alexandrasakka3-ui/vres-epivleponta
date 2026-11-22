#!/usr/bin/env python3
"""Semantic search over the unified thesis table."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DATA_DIR = Path(__file__).resolve().parent
PARQUET_PATH = DATA_DIR / "unified_clean.parquet"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
PEOPLE_PATH = DATA_DIR / "people.parquet"
EDGES_PATH = DATA_DIR / "edges.parquet"
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


def recommend_supervisors(
    results: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Rank supervisors represented in search results by result count and score."""
    if top_n < 0:
        raise ValueError("top_n must be non-negative.")
    required = {
        "supervisor_id",
        "supervisor_name",
        "institution",
        "title",
        "score",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Search results are missing columns: {sorted(missing)}")

    columns = [
        "supervisor_id",
        "supervisor_name",
        "institutions",
        "related_work_count",
        "score_sum",
        "top_titles",
    ]
    if top_n == 0 or results.empty:
        return pd.DataFrame(columns=columns)

    valid = results.loc[results["supervisor_id"].notna()].copy()
    valid["supervisor_id"] = valid["supervisor_id"].astype(str).str.strip()
    valid = valid.loc[valid["supervisor_id"] != ""]
    if valid.empty:
        return pd.DataFrame(columns=columns)

    valid["score"] = pd.to_numeric(valid["score"], errors="raise")
    recommendations = []
    for supervisor_id, group in valid.groupby("supervisor_id", sort=False):
        names = group["supervisor_name"].dropna().astype(str)
        name = next((value.strip() for value in names if value.strip()), "")
        institutions = sorted(
            {
                value.strip()
                for value in group["institution"].dropna().astype(str)
                if value.strip()
            }
        )
        ranked_works = group.sort_values("score", ascending=False, kind="stable")
        recommendations.append(
            {
                "supervisor_id": supervisor_id,
                "supervisor_name": name,
                "institutions": institutions,
                "related_work_count": len(group),
                "score_sum": group["score"].sum(),
                "top_titles": ranked_works["title"].head(3).tolist(),
            }
        )

    return (
        pd.DataFrame(recommendations, columns=columns)
        .sort_values(
            ["related_work_count", "score_sum", "supervisor_name"],
            ascending=[False, False, True],
            kind="stable",
        )
        .head(top_n)
        .reset_index(drop=True)
    )


@lru_cache(maxsize=1)
def _load_graph() -> tuple[pd.DataFrame, pd.DataFrame]:
    edges = pd.read_parquet(EDGES_PATH)
    people = pd.read_parquet(PEOPLE_PATH)
    required_edges = {"person_a", "person_b", "weight"}
    required_people = {"person_id", "canonical_name", "institutions"}
    missing_edges = required_edges.difference(edges.columns)
    missing_people = required_people.difference(people.columns)
    if missing_edges or missing_people:
        raise ValueError(
            f"Invalid graph data: missing edge columns {sorted(missing_edges)}, "
            f"missing people columns {sorted(missing_people)}."
        )
    return edges, people


def neighborhood(
    person_ids: Iterable[str] | str,
    max_nodes: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return people and induced edges around the strongest direct neighbors."""
    if max_nodes < 0:
        raise ValueError("max_nodes must be non-negative.")
    node_columns = ["person_id", "canonical_name", "institutions"]
    edge_columns = ["person_a", "person_b", "weight"]
    if max_nodes == 0:
        return pd.DataFrame(columns=node_columns), pd.DataFrame(columns=edge_columns)

    requested = [person_ids] if isinstance(person_ids, str) else person_ids
    seeds = []
    for person_id in requested:
        if person_id is None or pd.isna(person_id):
            continue
        normalized = str(person_id).strip()
        if normalized and normalized not in seeds:
            seeds.append(normalized)
    if not seeds:
        return pd.DataFrame(columns=node_columns), pd.DataFrame(columns=edge_columns)

    edges, people = _load_graph()
    people = people.drop_duplicates("person_id").copy()
    available_people = set(people["person_id"].dropna().astype(str))
    seeds = [person_id for person_id in seeds if person_id in available_people]
    if not seeds:
        return pd.DataFrame(columns=node_columns), edges.iloc[:0].copy()

    selected = seeds[:max_nodes]
    slots = max_nodes - len(selected)
    if slots:
        incident = edges.loc[
            edges["person_a"].isin(seeds) | edges["person_b"].isin(seeds)
        ]
        neighbor_weights: dict[str, int] = {}
        for edge in incident.itertuples(index=False):
            neighbor = (
                edge.person_b if edge.person_a in seeds else edge.person_a
            )
            if neighbor not in seeds:
                neighbor_weights[neighbor] = (
                    neighbor_weights.get(neighbor, 0) + int(edge.weight)
                )
        neighbors = sorted(
            neighbor_weights,
            key=lambda neighbor: (-neighbor_weights[neighbor], neighbor),
        )
        selected.extend(neighbors[:slots])

    selected_set = set(selected)
    nodes = people.loc[
        people["person_id"].isin(selected_set), node_columns
    ].copy()
    subgraph_edges = edges.loc[
        edges["person_a"].isin(selected_set)
        & edges["person_b"].isin(selected_set),
        edge_columns,
    ].copy()
    subgraph_edges = subgraph_edges.sort_values(
        ["weight", "person_a", "person_b"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    return nodes.reset_index(drop=True), subgraph_edges


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
