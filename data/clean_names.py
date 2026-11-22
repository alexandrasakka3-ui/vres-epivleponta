#!/usr/bin/env python3
"""Unify professor names in unified.parquet and write cleaned parquet files."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import OSA


DATA_DIR = Path(__file__).resolve().parent
INPUT_PATH = DATA_DIR / "unified.parquet"
CLEAN_PATH = DATA_DIR / "unified_clean.parquet"
PEOPLE_PATH = DATA_DIR / "people.parquet"
REPORT_PATH = DATA_DIR / "name_check.txt"

TITLE_PATTERNS = (
    re.compile(r"^(?:επ\.?\s*καθ\.?|αν\.?\s*καθ\.?)\s*", re.IGNORECASE),
    re.compile(r"^(?:καθ\.|δρ\.|prof(?:essor)?\.?|dr\.?)(?=\s|$)\s*", re.IGNORECASE),
)

GREEK_DIGRAPHS = {
    "ΕΙ": "I",
    "ΟΙ": "I",
    "ΟΥ": "OU",
    "ΜΠ": "B",
    "ΝΤ": "D",
    "ΓΚ": "G",
    "ΣΣ": "S",
}
GREEK_LETTERS = {
    "Α": "A",
    "Β": "V",
    "Γ": "G",
    "Δ": "D",
    "Ε": "E",
    "Ζ": "Z",
    "Η": "I",
    "Θ": "TH",
    "Ι": "I",
    "Κ": "K",
    "Λ": "L",
    "Μ": "M",
    "Ν": "N",
    "Ξ": "X",
    "Ο": "O",
    "Π": "P",
    "Ρ": "R",
    "Σ": "S",
    "Τ": "T",
    "Υ": "I",
    "Φ": "F",
    "Χ": "CH",
    "Ψ": "PS",
    "Ω": "O",
    "Ϲ": "S",
}
TOKEN_RE = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*\.?", re.UNICODE)


@dataclass
class NameRecord:
    surname: str
    given: str
    display_surname: str
    display_given: str
    raw_counts: Counter[str] = field(default_factory=Counter)
    supervisor_counts: Counter[str] = field(default_factory=Counter)

    @property
    def key(self) -> tuple[str, str]:
        return self.surname, self.given

    @property
    def is_initial(self) -> bool:
        return len(self.given) == 1

    @property
    def variants(self) -> set[str]:
        return set(self.raw_counts)


@dataclass
class PersonMetrics:
    n_as_supervisor: int = 0
    n_in_committee: int = 0
    institutions: set[str] = field(default_factory=set)


def _without_diacritics(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _strip_titles(value: str) -> str:
    value = " ".join(value.split())
    while True:
        for pattern in TITLE_PATTERNS:
            updated = pattern.sub("", value, count=1)
            if updated != value:
                value = updated.strip()
                break
        else:
            return value


def _phonetic_key(value: str) -> str:
    value = _without_diacritics(value).upper().replace("ς", "Σ")
    output: list[str] = []
    index = 0
    while index < len(value):
        pair = value[index : index + 2]
        if pair in GREEK_DIGRAPHS:
            output.append(GREEK_DIGRAPHS[pair])
            index += 2
            continue
        char = value[index]
        output.append(GREEK_LETTERS.get(char, char))
        index += 1
    key = "".join(char for char in output if char.isalnum())
    return re.sub(r"(.)\1+", r"\1", key)


def _matching_text(value: str) -> str:
    return _phonetic_key(value)


def _is_greek_text(value: str) -> bool:
    return any("GREEK" in unicodedata.name(char, "") for char in value)


def _pretty(value: str) -> str:
    def capitalize(match: re.Match[str]) -> str:
        word = match.group(0)
        return word[:1].upper() + word[1:].lower()

    return re.sub(r"[^\W\d_]+", capitalize, value, flags=re.UNICODE)


def _is_initial(value: str) -> bool:
    return len(_matching_text(value)) == 1


def _parse_name(raw_name: str) -> tuple[str, str, str, str] | None:
    cleaned = _strip_titles(raw_name)
    if not cleaned:
        return None

    if "," in cleaned:
        surname_text, given_text = cleaned.split(",", maxsplit=1)
        surname_tokens = TOKEN_RE.findall(surname_text.strip())
        given_tokens = TOKEN_RE.findall(given_text.strip())
        surname = " ".join(surname_tokens).strip()
        given = " ".join(given_tokens).strip()
    else:
        tokens = TOKEN_RE.findall(cleaned)
        if not tokens:
            return None
        if len(tokens) == 1:
            surname, given = tokens[0], ""
        elif _is_initial(tokens[0]):
            given = tokens[0]
            surname = " ".join(tokens[1:])
        elif _is_initial(tokens[-1]):
            surname = " ".join(tokens[:-1])
            given = tokens[-1]
        else:
            surname = tokens[0]
            given = " ".join(tokens[1:])

    if not surname:
        return None
    surname_display = _pretty(surname)
    given_display = _pretty(given)
    return (
        _matching_text(surname),
        _matching_text(given.split(maxsplit=1)[0]) if given else "",
        surname_display,
        given_display,
    )


def _supervisor_name(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.split(";", maxsplit=1)[0].strip()


def _format_name(record: NameRecord) -> str:
    if record.display_given:
        return f"{record.display_surname}, {record.display_given}"
    return record.display_surname


def _gendered_surname_match(first: str, second: str) -> bool:
    endings = (("IS", "I"), ("OS", "OU"))
    return any(
        first.endswith(masculine)
        and second.endswith(feminine)
        and first[: -len(masculine)] == second[: -len(feminine)]
        or second.endswith(masculine)
        and first.endswith(feminine)
        and second[: -len(masculine)] == first[: -len(feminine)]
        for masculine, feminine in endings
    )


def _surname_relation(first: NameRecord, second: NameRecord) -> str | None:
    if first.surname == second.surname:
        return "normalized"
    if _gendered_surname_match(first.surname, second.surname):
        if (
            first.given
            and second.given
            and not first.is_initial
            and not second.is_initial
            and first.given == second.given
        ):
            return "gender"
        return None
    if (
        OSA.distance(first.surname, second.surname) <= 1
        and first.given
        and second.given
        and not first.is_initial
        and not second.is_initial
        and fuzz.ratio(first.given, second.given) >= 95
    ):
        return "fuzzy"
    return None


def _full_names_compatible(
    anchor: NameRecord, candidate: NameRecord, surname_relation: str
) -> bool:
    if (
        not anchor.given
        or not candidate.given
        or anchor.is_initial
        or candidate.is_initial
    ):
        return False
    ratio = fuzz.ratio(anchor.given, candidate.given)
    if surname_relation == "fuzzy":
        return ratio >= 95
    if surname_relation == "gender":
        return anchor.given == candidate.given
    return ratio >= 85


def _canonical_name(records: list[NameRecord]) -> str:
    candidates: Counter[str] = Counter()
    supervisor_candidates: Counter[str] = Counter()
    for record in records:
        for variant, count in record.raw_counts.items():
            parsed = _parse_name(variant)
            if parsed is None:
                continue
            _, given_key, surname_display, given_display = parsed
            if not given_key or len(given_key) == 1:
                continue
            display = (
                f"{surname_display}, {given_display}"
                if given_display
                else surname_display
            )
            candidates[display] += count
            supervisor_candidates[display] += record.supervisor_counts[variant]

    def rank(candidate: str) -> tuple[int, int, int, str]:
        greek_preference = int(_is_greek_text(candidate))
        return (
            -greek_preference,
            -candidates[candidate],
            -supervisor_candidates[candidate],
            candidate,
        )

    return min(candidates, key=rank) if candidates else _format_name(records[0])


def _collect_names(
    rows: list[dict[str, object]],
) -> tuple[list[NameRecord], dict[str, int], Counter[str]]:
    raw_counts: Counter[str] = Counter()
    supervisor_counts: Counter[str] = Counter()
    for row in rows:
        supervisor = _supervisor_name(row.get("supervisor"))
        if supervisor:
            raw_counts[supervisor] += 1
            supervisor_counts[supervisor] += 1
        committee = row.get("committee") or []
        for name in committee:
            if isinstance(name, str) and name.strip():
                raw_counts[name] += 1

    by_key: dict[tuple[str, str], NameRecord] = {}
    raw_to_index: dict[str, int] = {}
    for raw_name, count in raw_counts.items():
        parsed = _parse_name(raw_name)
        if parsed is None:
            raise ValueError(f"Could not parse non-empty name: {raw_name!r}")
        surname, given, display_surname, display_given = parsed
        key = (surname, given)
        record = by_key.get(key)
        if record is None:
            record = NameRecord(surname, given, display_surname, display_given)
            by_key[key] = record
        record.raw_counts[raw_name] += count
        record.supervisor_counts[raw_name] += supervisor_counts[raw_name]

    records = sorted(by_key.values(), key=lambda item: (item.surname, item.given))
    key_to_index = {record.key: index for index, record in enumerate(records)}
    for raw_name in raw_counts:
        parsed = _parse_name(raw_name)
        if parsed is None:
            raise ValueError(f"Could not parse non-empty name: {raw_name!r}")
        raw_to_index[raw_name] = key_to_index[(parsed[0], parsed[1])]
    return records, raw_to_index, raw_counts


def _group_names(records: list[NameRecord]) -> list[list[int]]:
    full_names = [
        index
        for index, record in enumerate(records)
        if record.given and not record.is_initial
    ]
    full_names.sort(
        key=lambda index: (
            -sum(records[index].raw_counts.values()),
            records[index].surname,
            records[index].given,
        )
    )

    groups: list[list[int]] = []
    anchors: list[int] = []
    for candidate_index in full_names:
        candidate = records[candidate_index]
        compatible_anchors: list[tuple[float, int, int]] = []
        for group_index, anchor_index in enumerate(anchors):
            anchor = records[anchor_index]
            relation = _surname_relation(anchor, candidate)
            if relation is None or not _full_names_compatible(
                anchor, candidate, relation
            ):
                continue
            compatible_anchors.append(
                (fuzz.ratio(anchor.given, candidate.given), -group_index, group_index)
            )
        if compatible_anchors:
            group_index = max(compatible_anchors)[2]
            groups[group_index].append(candidate_index)
        else:
            anchors.append(candidate_index)
            groups.append([candidate_index])

    anchors_for_initial: dict[tuple[str, str], list[int]] = defaultdict(list)
    for group_index, anchor_index in enumerate(anchors):
        anchor = records[anchor_index]
        anchors_for_initial[(anchor.surname, anchor.given[0])].append(group_index)

    unassigned: list[int] = []
    for index, record in enumerate(records):
        if record.given and not record.is_initial:
            continue
        if record.is_initial:
            matching_groups = anchors_for_initial.get(
                (record.surname, record.given), []
            )
            if len(matching_groups) == 1:
                groups[matching_groups[0]].append(index)
                continue
        unassigned.append(index)

    groups.extend([[index] for index in unassigned])
    return sorted(groups, key=lambda group: min(records[index].key for index in group))


def main() -> None:
    table = pq.read_table(INPUT_PATH)
    required = {"supervisor", "committee", "institution"}
    missing = required.difference(table.column_names)
    if missing:
        raise ValueError(f"{INPUT_PATH} is missing required columns: {sorted(missing)}")

    rows = table.to_pylist()
    records, raw_to_index, raw_counts = _collect_names(rows)
    groups = _group_names(records)

    group_for_index: dict[int, int] = {}
    for group_index, group in enumerate(groups):
        for record_index in group:
            group_for_index[record_index] = group_index

    canonical_names = [
        _canonical_name([records[index] for index in group]) for group in groups
    ]
    sort_order = sorted(
        range(len(groups)),
        key=lambda index: (
            _matching_text(canonical_names[index]),
            canonical_names[index],
        ),
    )
    person_ids = {
        group_index: f"P{position:06d}"
        for position, group_index in enumerate(sort_order, 1)
    }
    id_for_raw = {
        raw_name: person_ids[group_for_index[record_index]]
        for raw_name, record_index in raw_to_index.items()
    }

    def canonical_for_raw(raw_name: object) -> str | None:
        if not isinstance(raw_name, str) or not raw_name.strip():
            return None
        record_index = raw_to_index.get(raw_name)
        if record_index is None:
            return None
        group_index = group_for_index[record_index]
        return canonical_names[group_index]

    supervisor_ids: list[str | None] = []
    supervisor_names: list[str | None] = []
    committee_ids: list[list[str]] = []
    committee_names: list[list[str]] = []
    people_metrics = {index: PersonMetrics() for index in range(len(groups))}
    for row in rows:
        supervisor = _supervisor_name(row.get("supervisor"))
        supervisor_id = (
            id_for_raw.get(supervisor)
            if supervisor
            else None
        )
        supervisor_ids.append(supervisor_id)
        supervisor_names.append(canonical_for_raw(supervisor))
        committee = row.get("committee") or []
        row_committee_ids: list[str] = []
        row_committee_names: list[str] = []
        for name in committee:
            if not isinstance(name, str) or not name.strip():
                continue
            name_id = id_for_raw[name]
            row_committee_ids.append(name_id)
            row_committee_names.append(canonical_for_raw(name) or "")

        committee_ids.append(row_committee_ids)
        committee_names.append(row_committee_names)
        institution = row.get("institution")
        if supervisor_id is not None:
            group_index = group_for_index[raw_to_index[supervisor]]
            people_metrics[group_index].n_as_supervisor += 1
            if institution:
                people_metrics[group_index].institutions.add(str(institution))
        for name in committee:
            if not isinstance(name, str) or not name.strip():
                continue
            group_index = group_for_index[raw_to_index[name]]
            people_metrics[group_index].n_in_committee += 1
            if institution:
                people_metrics[group_index].institutions.add(str(institution))

    clean_table = table.append_column(
        "supervisor_id", pa.array(supervisor_ids, type=pa.string())
    )
    clean_table = clean_table.append_column(
        "supervisor_name", pa.array(supervisor_names, type=pa.string())
    )
    clean_table = clean_table.append_column(
        "committee_ids", pa.array(committee_ids, type=pa.list_(pa.string()))
    )
    clean_table = clean_table.append_column(
        "committee_names", pa.array(committee_names, type=pa.list_(pa.string()))
    )
    pq.write_table(clean_table, CLEAN_PATH, compression="zstd")

    people_rows: list[dict[str, object]] = []
    for group_index, group in enumerate(groups):
        variants = sorted(
            {
                raw_name
                for record_index in group
                for raw_name in records[record_index].raw_counts
            },
            key=lambda name: (-raw_counts[name], name),
        )
        metric = people_metrics[group_index]
        people_rows.append(
            {
                "person_id": person_ids[group_index],
                "canonical_name": canonical_names[group_index],
                "variants": variants,
                "n_as_supervisor": metric.n_as_supervisor,
                "n_in_committee": metric.n_in_committee,
                "institutions": sorted(metric.institutions),
            }
        )
    people_schema = pa.schema(
        [
            ("person_id", pa.string()),
            ("canonical_name", pa.string()),
            ("variants", pa.list_(pa.string())),
            ("n_as_supervisor", pa.int64()),
            ("n_in_committee", pa.int64()),
            ("institutions", pa.list_(pa.string())),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(people_rows, schema=people_schema),
        PEOPLE_PATH,
        compression="zstd",
    )

    group_reports = sorted(
        enumerate(groups),
        key=lambda item: (
            -sum(len(records[index].variants) for index in item[1]),
            -sum(sum(records[index].raw_counts.values()) for index in item[1]),
            canonical_names[item[0]],
        ),
    )
    most_frequent_supervisors = Counter(
        group_for_index[
            raw_to_index[_supervisor_name(row.get("supervisor"))]
        ]
        for row in rows
        if _supervisor_name(row.get("supervisor"))
    ).most_common(20)
    report_lines = [
        "Ενοποίηση ονομάτων καθηγητών",
        "==============================",
        f"Διαφορετικές μη κενές γραφές πριν: {len(raw_counts):,}",
        f"Διαφορετικές canonical γραφές/ομάδες μετά: {len(groups):,}",
        f"Γραμμές εισόδου: {len(rows):,}",
        "Οι αρχικές στήλες και η σειρά γραμμών του unified.parquet διατηρούνται στο unified_clean.parquet.",
        "",
        "Υποθέσεις και κανόνες",
        "---------------------",
        "- Αφαιρούνται μόνο αρχικοί τίτλοι Καθ., Επ. Καθ., Αν. Καθ., Δρ., Prof/Professor και Dr.",
        "- Το φωνητικό κλειδί αφαιρεί τόνους/διαλυτικά, εξισώνει ελληνικά/λατινικά και συμπτύσσει διπλά γράμματα.",
        "- Με κόμμα, το αριστερό τμήμα είναι επώνυμο. Χωρίς κόμμα, πρώτο token επώνυμο, εκτός αν πρώτο ή τελευταίο token είναι αρχικό.",
        "- Τα πλήρη ονόματα ταξινομούνται κατά συχνότητα και ομαδοποιούνται μόνο γύρω από anchor (αστεροειδώς), με ratio μικρού ονόματος ≥85.",
        "- Αρχικά ανατίθενται μόνο όταν υπάρχει ακριβώς ένα anchor με το ίδιο κανονικοποιημένο επώνυμο και αρχικό.",
        "- Επώνυμα ενώνονται με ίδιο φωνητικό κλειδί· OSA ≤1 απαιτεί πλήρες μικρό όνομα και ratio ≥95. Καταλήξεις -ης/-η και -ος/-ου απαιτούν ίδιο πλήρες μικρό όνομα.",
        "- Canonical είναι η συχνότερη ελληνική πλήρης γραφή· αν δεν υπάρχει, η συχνότερη λατινική. Σε ισοπαλία προτιμάται γραφή που εμφανίζεται ως supervisor.",
        "- Σε τιμές supervisor με πολλά ονόματα χωρισμένα με «;» κρατιέται μόνο το πρώτο, όπως ζητήθηκε.",
        "",
        "20 μεγαλύτερες ομάδες (όλες οι αρχικές παραλλαγές)",
        "-------------------------------------------------",
    ]
    for position, (group_index, group) in enumerate(group_reports[:20], start=1):
        variants = sorted(
            {
                raw_name
                for record_index in group
                for raw_name in records[record_index].raw_counts
            },
            key=lambda name: (-raw_counts[name], name),
        )
        count = sum(raw_counts[name] for name in variants)
        report_lines.append(
            f"{position}. {canonical_names[group_index]} — "
            f"{len(variants)} παραλλαγές, {count} εμφανίσεις"
        )
        report_lines.extend(f"   - {variant}" for variant in variants)

    conflicting_groups: list[tuple[int, list[int], float]] = []
    for group_index, group in group_reports:
        full_records = [
            records[index]
            for index in group
            if records[index].given and not records[index].is_initial
        ]
        minimum_ratio = 100.0
        for first_pos, first in enumerate(full_records):
            for second in full_records[first_pos + 1 :]:
                minimum_ratio = min(
                    minimum_ratio, fuzz.ratio(first.given, second.given)
                )
        if minimum_ratio < 85:
            conflicting_groups.append((group_index, group, minimum_ratio))

    report_lines.extend(
        [
            "",
            "30 ομάδες με σύγκρουση (πλήρη μικρά ονόματα με ratio <85)",
            "---------------------------------------------------------",
        ]
    )
    for position, (group_index, group, minimum_ratio) in enumerate(
        conflicting_groups[:30], start=1
    ):
        variants = sorted(
            {
                raw_name
                for record_index in group
                for raw_name in records[record_index].raw_counts
            },
            key=lambda name: (-raw_counts[name], name),
        )
        report_lines.append(
            f"{position}. {canonical_names[group_index]} — "
            f"χαμηλότερο ratio μικρών ονομάτων {minimum_ratio:.1f}"
        )
        report_lines.extend(f"   - {variant}" for variant in variants)
    if not conflicting_groups:
        report_lines.append("- Δεν βρέθηκαν ομάδες με τέτοια σύγκρουση.")

    report_lines.extend(["", "20 συχνότεροι επιβλέποντες", "--------------------------"])
    for position, (group_index, count) in enumerate(
        most_frequent_supervisors, start=1
    ):
        report_lines.append(
            f"{position}. {canonical_names[group_index]} — {count}"
        )

    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(
        f"Processed {len(rows):,} rows and {len(raw_counts):,} distinct names "
        f"into {len(groups):,} people."
    )
    print(f"Wrote {CLEAN_PATH}, {PEOPLE_PATH}, and {REPORT_PATH}.")


if __name__ == "__main__":
    main()
