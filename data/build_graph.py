#!/usr/bin/env python3
"""Build a weighted collaboration graph from thesis supervisors and committees."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
UNIFIED_PATH = DATA_DIR / "unified_clean.parquet"
PEOPLE_PATH = DATA_DIR / "people.parquet"
EDGES_PATH = DATA_DIR / "edges.parquet"
MAX_PEOPLE_PER_WORK = 12


def _person_id(value: object) -> str | None:
    if value is None or (not isinstance(value, (list, tuple, np.ndarray)) and pd.isna(value)):
        return None
    normalized = str(value).strip()
    return normalized or None


def _work_people(supervisor_id: object, committee_ids: object) -> list[str]:
    person_ids = set()
    supervisor = _person_id(supervisor_id)
    if supervisor is not None:
        person_ids.add(supervisor)

    if committee_ids is None or (
        not isinstance(committee_ids, (list, tuple, np.ndarray))
        and pd.isna(committee_ids)
    ):
        committee_values = ()
    elif isinstance(committee_ids, (list, tuple, np.ndarray)):
        committee_values = committee_ids
    else:
        committee_values = (committee_ids,)

    person_ids.update(
        person_id
        for value in committee_values
        if (person_id := _person_id(value)) is not None
    )
    return sorted(person_ids)


def build_graph() -> tuple[pd.DataFrame, int]:
    frame = pd.read_parquet(
        UNIFIED_PATH,
        columns=["supervisor_id", "committee_ids"],
    )
    weights: Counter[tuple[str, str]] = Counter()
    skipped_works = 0

    for row in frame.itertuples(index=False):
        person_ids = _work_people(row.supervisor_id, row.committee_ids)
        if len(person_ids) > MAX_PEOPLE_PER_WORK:
            skipped_works += 1
            continue
        weights.update(combinations(person_ids, 2))

    edges = pd.DataFrame(
        [
            {"person_a": person_a, "person_b": person_b, "weight": weight}
            for (person_a, person_b), weight in weights.items()
        ],
        columns=["person_a", "person_b", "weight"],
    )
    if not edges.empty:
        edges = edges.sort_values(
            ["person_a", "person_b"], kind="stable"
        ).reset_index(drop=True)
        if not (edges["person_a"] < edges["person_b"]).all():
            raise ValueError("Graph edges must have person_a < person_b.")
        edges["weight"] = edges["weight"].astype("int64")
    else:
        edges = edges.astype(
            {"person_a": "string", "person_b": "string", "weight": "int64"}
        )

    edges.to_parquet(EDGES_PATH, index=False)
    return edges, skipped_works


def main() -> None:
    edges, skipped_works = build_graph()
    people = pd.read_parquet(PEOPLE_PATH, columns=["person_id", "canonical_name"])
    names = dict(zip(people["person_id"], people["canonical_name"]))
    connected_people = set(edges["person_a"]) | set(edges["person_b"])
    print(f"Εργασίες που αγνοήθηκαν (>12 άτομα): {skipped_works:,}")
    print(f"Κόμβοι με τουλάχιστον μία ακμή: {len(connected_people):,}")
    print(f"Ακμές: {len(edges):,}")
    print(f"Μέγεθος αρχείου: {EDGES_PATH.stat().st_size:,} bytes")
    print("10 ισχυρότερες ακμές:")
    strongest = edges.sort_values(
        ["weight", "person_a", "person_b"],
        ascending=[False, True, True],
        kind="stable",
    ).head(10)
    for edge in strongest.itertuples(index=False):
        name_a = names.get(edge.person_a) or edge.person_a
        name_b = names.get(edge.person_b) or edge.person_b
        print(f"  {name_a} — {name_b}: {edge.weight}")


if __name__ == "__main__":
    main()
