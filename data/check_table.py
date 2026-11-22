import random
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


INSTITUTIONS = ("Θράκη", "Κρήτη", "Μακεδονία")
SOURCE_FILES = {
    "elocus": ("glossAPI/elocus", "elocus.parquet"),
    "psepheda": ("glossAPI/psepheda", "psepheda.parquet"),
}


def clean_text(value):
    if value is None:
        return ""
    return " ".join(str(value).split())


def split_names(value, separator):
    return [
        name
        for part in (value or "").split(separator)
        if (name := clean_text(part))
    ]


def elocus_department(collection):
    sections = clean_text(collection).split("--")
    departments = [
        section for section in sections
        if "τμήμα" in section.casefold()
    ]
    return clean_text(departments[-1]) if departments else ""


def psepheda_has_department_entry(entries, source_department):
    if not entries:
        return False
    last_entry = entries[-1]
    normalized_department = clean_text(source_department).casefold()
    return (
        last_entry.casefold().startswith("πανεπιστήμιο")
        or "τμήμα" in last_entry.casefold()
        or bool(normalized_department and last_entry.casefold() == normalized_department)
    )


def committee_size(count):
    if count >= 4:
        return "4+"
    return str(count)


def raw_value(row, column):
    if column not in row:
        return "<η στήλη δεν υπάρχει>"
    value = row[column]
    if value is None:
        return "<None>"
    return repr(str(value))


def read_pylist(parquet_file, columns):
    for batch in parquet_file.iter_batches(
        batch_size=512,
        columns=columns,
        use_threads=True,
    ):
        yield from pa.Table.from_batches([batch]).to_pylist()


def main():
    data_dir = Path(__file__).resolve().parent
    unified_path = data_dir / "unified.parquet"
    unified = pq.read_table(unified_path).to_pylist()

    output = []
    output.append("Έλεγχος data/unified.parquet")
    output.append(f"Συνολικές γραμμές: {len(unified):,}")
    output.append("")

    rng = random.Random(20261122)
    for institution in INSTITUTIONS:
        rows = [row for row in unified if row["institution"] == institution]
        output.append(f"{institution}")
        output.append("-" * len(institution))
        output.append("1. 6 τυχαίες γραμμές:")

        samples = rng.sample(rows, min(6, len(rows)))
        for index, row in enumerate(samples, start=1):
            committee = row["committee"] if row["committee"] is not None else []
            output.append(
                f"  {index}. supervisor={repr(row['supervisor'])}; "
                f"committee={committee!r}; department={repr(row['department'])}"
            )

        sizes = {"0": 0, "1": 0, "2": 0, "3": 0, "4+": 0}
        for row in rows:
            committee = row["committee"] or []
            sizes[committee_size(len(committee))] += 1

        output.append("2. Κατανομή αριθμού ονομάτων στο committee:")
        for size in ("0", "1", "2", "3", "4+"):
            output.append(f"  {size}: {sizes[size]:,}")
        output.append("")

    output.append("Κρήτη και Μακεδονία — ακριβής λογική committee")
    output.append(
        "  Κρήτη: οι στήλες των αρχικών δεδομένων "
        "advisor και committee_members."
    )
    output.append(
        "  Η build_table.py διαβάζει το advisor, χωρίζει την τιμή με αλλαγή "
        "γραμμής («\\n»), βάζει το πρώτο όνομα στο supervisor και τα υπόλοιπα "
        "στο committee· έπειτα προσθέτει τα ονόματα από το committee_members "
        "χωρισμένα με ερωτηματικό («;»)."
    )
    output.append(
        "  Μακεδονία: η στήλη των αρχικών δεδομένων contributors."
    )
    output.append(
        "  Η build_table.py τη χωρίζει με ερωτηματικό («;») και βάζει το πρώτο "
        "στο supervisor. Τα επόμενα στοιχεία είναι committee, εκτός από το "
        "τελευταίο όταν είναι αναγνωρίσιμο ως τμήμα/πανεπιστήμιο."
    )
    output.append("")

    paths = {}
    for name, (repo_id, filename) in SOURCE_FILES.items():
        paths[name] = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
        )

    elocus_file = pq.ParquetFile(paths["elocus"])
    elocus_columns = [
        "advisor",
        "committee_members",
        "supervisor",
        "contributors",
        "url",
    ]
    elocus_columns = [
        column for column in elocus_columns
        if column in elocus_file.schema_arrow.names
    ]
    elocus_rows = {}
    for row in read_pylist(elocus_file, elocus_columns):
        elocus_rows.setdefault(clean_text(row.get("url")), []).append(row)

    output.append("Κρήτη — 6 παραδείγματα με κενό committee:")
    crete_examples = 0
    for row in unified:
        if row["institution"] != "Κρήτη" or (row["committee"] or []):
            continue
        matches = elocus_rows.get(clean_text(row["url"]), [])
        if not matches:
            continue
        original = matches[0]
        output.append(f"  Παράδειγμα {crete_examples + 1}:")
        output.append(f"    contributors={raw_value(original, 'contributors')}")
        output.append(f"    advisor={raw_value(original, 'advisor')}")
        output.append(f"    supervisor={raw_value(original, 'supervisor')}")
        output.append(
            f"    committee_members={raw_value(original, 'committee_members')}"
        )
        crete_examples += 1
        if crete_examples == 6:
            break
    if crete_examples < 6:
        output.append(f"  Βρέθηκαν μόνο {crete_examples} παραδείγματα.")
    output.append("")

    psepheda_file = pq.ParquetFile(paths["psepheda"])
    psepheda_columns = [
        "supervisor",
        "contributors",
        "department",
        "abstract",
        "issue_date",
        "handle_url",
        "title",
    ]
    psepheda_columns = [
        column for column in psepheda_columns
        if column in psepheda_file.schema_arrow.names
    ]

    macedonia_empty_urls = {
        clean_text(row["url"])
        for row in unified
        if row["institution"] == "Μακεδονία"
        and not (row["committee"] or [])
    }
    macedonia_examples = 0
    psepheda_rows = []
    output.append("Μακεδονία — 6 παραδείγματα με κενό committee:")
    for row in read_pylist(psepheda_file, psepheda_columns):
        psepheda_rows.append(row)
        if clean_text(row.get("handle_url")) not in macedonia_empty_urls:
            continue
        supervisor = clean_text(row.get("supervisor"))
        contributors = split_names(row.get("contributors"), ";")
        has_department_entry = psepheda_has_department_entry(
            contributors, row.get("department")
        )
        committee = contributors[1:]
        if has_department_entry:
            committee = contributors[1:-1]

        abstract = clean_text(row.get("abstract"))
        record_is_included = len(abstract) >= 50 and bool(supervisor)
        matching_unified = (
            record_is_included
            and not committee
            and macedonia_examples < 6
        )
        if matching_unified:
            output.append(f"  Παράδειγμα {macedonia_examples + 1}:")
            output.append(f"    contributors={raw_value(row, 'contributors')}")
            output.append(f"    advisor={raw_value(row, 'advisor')}")
            output.append(f"    supervisor={raw_value(row, 'supervisor')}")
            macedonia_examples += 1
    if macedonia_examples < 6:
        output.append(f"  Βρέθηκαν μόνο {macedonia_examples} παραδείγματα.")
    output.append("")

    qualifying = 0
    qualifying_nonempty_committee = 0
    for row in psepheda_rows:
        contributors = split_names(row.get("contributors"), ";")
        if len(contributors) <= 2:
            continue
        qualifying += 1

        abstract = clean_text(row.get("abstract"))
        supervisor = clean_text(row.get("supervisor"))
        committee = contributors[1:]
        if psepheda_has_department_entry(contributors, row.get("department")):
            committee = contributors[1:-1]
        if len(abstract) >= 50 and supervisor and committee:
            qualifying_nonempty_committee += 1

    output.append("psepheda")
    output.append(
        "  Αριθμός γραμμών του αρχικού αρχείου με >2 στοιχεία στο "
        "contributors (χωρισμένα με «;», κενά στοιχεία δεν μετράνε): "
        f"{qualifying:,}"
    )
    output.append(
        "  Από αυτές, γραμμές που κατέληξαν με μη κενό committee: "
        f"{qualifying_nonempty_committee:,}"
    )
    output.append("")

    output.append("Σημείωση για πιθανό λάθος στη λογική")
    output.append(
        "  Η ρητή λογική κόβει το τελευταίο στοιχείο του contributors ως "
        "τμήμα/πανεπιστήμιο όταν ναι, αλλά το committee μπορεί να ορίσει λάθος "
        "όταν τα δεδομένα περιέχουν τμήμα/πανεπιστήμιο αλλά υπάρχουν επιπλέον "
        "μέλη, ή όταν το τελευταίο πραγματικό όνομα μοιάζει με ορολογία "
        "τμήματος."
    )
    output.append(
        "  Επιπλέον, το psepheda θεωρεί το πρώτο στοιχείο του contributors "
        "πάντα supervisor, οπότε η πραγματική επιβλέπουσα μπορεί να αγνοείται "
        "αν το πρώτο στοιχείο είναι μέλος επιτροπής ή οργανωτική εγγραφή."
    )
    output.append(
        "  Προτεινόμενη διόρθωση: διαχωρίστε σαφώς supervisor, committee και "
        "department ανά πεδίο όπου είναι διαθέσιμο· όπου δεν είναι, δημιουργήστε "
        "και δοκιμάστε έναν κανονικοποιημένο κανόνα με ονομαστική λίστα και "
        "ξεχωριστή ταξινόμηση οργανισμών, και μην αφαιρείτε ποτέ το τελευταίο "
        "στοιχείο από blind-heuristic χωρίς επιβεβαίωση."
    )

    text = "\n".join(output) + "\n"
    (data_dir / "check_table.txt").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
