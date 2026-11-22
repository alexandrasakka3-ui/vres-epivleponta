from collections import Counter
from pathlib import Path
import signal
import traceback

import duckdb
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


DATASETS = {
    "libduth": ("glossAPI/libduth", "libduth.parquet"),
    "elocus": ("glossAPI/elocus", "elocus.parquet"),
    "psepheda": ("glossAPI/psepheda", "psepheda.parquet"),
}
CONTENT_TERMS = ("επιτροπ", "Τριμελ", "Εξεταστ")
ORGANIZATION_TERMS = (
    "πανεπιστήμιο",
    "τμήμα",
    "σχολή",
    "ινστιτούτο",
    "εργαστήριο",
    "κέντρο",
    "πρόγραμμα",
    "διατμηματικό",
    "ανώτατη βιομηχανική σχολή",
)


def download(dataset):
    repo_id, filename = DATASETS[dataset]
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
    )


def text_value(value):
    return "" if value is None else str(value)


def inspect_psepheda(path, output):
    parquet_file = pq.ParquetFile(path)
    column_names = parquet_file.schema_arrow.names
    requested_columns = ["supervisor", "contributors"]
    missing = [column for column in requested_columns if column not in column_names]
    if missing:
        raise ValueError(f"psepheda missing required columns: {', '.join(missing)}")

    contributor_values = []
    row_count = 0
    for batch in parquet_file.iter_batches(
        batch_size=256,
        columns=requested_columns,
        use_threads=True,
    ):
        contributors = batch.column(
            batch.schema.get_field_index("contributors")
        ).to_pylist()
        row_count += len(contributors)
        contributor_values.extend(contributors)

    nonempty = [text_value(value).strip() for value in contributor_values]
    nonempty = [value for value in nonempty if value]
    element_counts = Counter()
    total_elements = 0
    for value in contributor_values:
        elements = [part.strip() for part in text_value(value).split(";")]
        elements = [part for part in elements if part]
        total_elements += len(elements)
        element_counts.update(elements)

    output.append("Μέρος Α — psepheda (μόνο supervisor, contributors)")
    output.append(f"Μη κενές τιμές contributors: {len(nonempty)}/{row_count}")
    output.append("15 πρώτες μη κενές τιμές contributors (ολόκληρες):")
    for index, value in enumerate(nonempty[:15], start=1):
        output.append(f"  {index}. {value}")
    output.append(
        "Μέσος αριθμός στοιχείων ανά γραμμή "
        f"(διαχωρισμός με «;», κενές γραμμές ως 0): "
        f"{total_elements / row_count:.2f}"
        if row_count
        else "Μέσος αριθμός στοιχείων ανά γραμμή: δεν υπάρχουν γραμμές"
    )
    output.append("25 συχνότερα στοιχεία contributors:")
    most_common = element_counts.most_common(25)
    for index, (element, count) in enumerate(most_common, start=1):
        output.append(f"  {index}. {element} — {count}")
    organization_count = sum(
        any(term in element.casefold() for term in ORGANIZATION_TERMS)
        for element, _ in most_common
    )
    person_count = len(most_common) - organization_count
    output.append(
        "Εκτίμηση για τα 25 συχνότερα: "
        f"{person_count} μοιάζουν με ονόματα προσώπων και "
        f"{organization_count} με ονόματα προγραμμάτων/φορέων/τμημάτων/σχολών."
    )


def inspect_contributors_column(dataset, path, output):
    parquet_file = pq.ParquetFile(path)
    if "contributors" not in parquet_file.schema_arrow.names:
        output.append(f"{dataset}: δεν υπάρχει στήλη contributors")
        return

    total = 0
    nonempty = 0
    for batch in parquet_file.iter_batches(
        batch_size=512,
        columns=["contributors"],
        use_threads=True,
    ):
        values = batch.column(
            batch.schema.get_field_index("contributors")
        ).to_pylist()
        total += len(values)
        nonempty += sum(bool(text_value(value).strip()) for value in values)
    percentage = 100 * nonempty / total if total else 0
    output.append(
        f"{dataset}: contributors μη κενό σε {nonempty}/{total} γραμμές "
        f"({percentage:.2f}%)"
    )


def inspect_content(dataset, path, output):
    escaped_path = str(path).replace("'", "''")
    query = f"SELECT substr(content, 1, 6000) FROM '{escaped_path}' LIMIT 300"
    output.append(f"{dataset} — DuckDB content (πρώτοι έως 300 × 6000 χαρακτήρες):")
    print(f"Running DuckDB content query for {dataset}...", flush=True)
    try:
        with duckdb.connect() as connection:
            connection.execute("PRAGMA memory_limit='2GB'")
            connection.execute("SET threads=1")
            rows = connection.execute(query).fetchall()
    except Exception as error:
        output.append("  Το ερώτημα απέτυχε.")
        output.append(f"  Ακριβές σφάλμα: {type(error).__name__}: {error}")
        output.append("  Traceback:")
        output.extend(f"    {line}" for line in traceback.format_exc().rstrip().splitlines())
        return

    print(f"Finished DuckDB content query for {dataset}: {len(rows)} rows.", flush=True)
    texts = [text_value(row[0]) for row in rows]
    matches = []
    for row_number, text in enumerate(texts, start=1):
        folded = text.casefold()
        found = [
            index
            for term in CONTENT_TERMS
            if (index := folded.find(term.casefold())) >= 0
        ]
        if found:
            matches.append((row_number, text, min(found)))

    total = len(texts)
    percentage = 100 * len(matches) / total if total else 0
    output.append(
        f"  Κείμενα με μία από τις λέξεις "
        f"{', '.join(CONTENT_TERMS)}: {len(matches)}/{total} ({percentage:.2f}%)"
    )
    output.append("  Παραδείγματα (έως 3, 400 χαρακτήρες γύρω από την πρώτη εμφάνιση):")
    if not matches:
        output.append("    Δεν βρέθηκαν παραδείγματα.")
    for row_number, text, first_index in matches[:3]:
        start = min(max(first_index - 200, 0), max(len(text) - 400, 0))
        excerpt = text[start : start + 400]
        output.append(f"    Row {row_number}: {excerpt}")


def main():
    output = []
    output_path = Path(__file__).resolve().parent.parent / "committee_check2.txt"
    operation = "dataset processing"

    def save_output():
        output_path.write_text("\n".join(output) + "\n", encoding="utf-8")

    def handle_sigterm(signum, _frame):
        output.append(
            f"{operation}: process terminated by signal "
            f"{signal.Signals(signum).name} ({signum}); exit status {128 + signum}. "
            "No DuckDB exception was surfaced."
        )
        save_output()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_sigterm)
    paths = {}
    for dataset in DATASETS:
        print(f"Downloading/resolving {dataset}...", flush=True)
        try:
            paths[dataset] = download(dataset)
            print(f"Ready: {dataset}", flush=True)
        except Exception as error:
            output.append(f"{dataset}: αποτυχία λήψης parquet")
            output.append(f"  Ακριβές σφάλμα: {type(error).__name__}: {error}")
            output.append("  Traceback:")
            output.extend(
                f"    {line}"
                for line in traceback.format_exc().rstrip().splitlines()
            )

    if "psepheda" in paths:
        try:
            inspect_psepheda(paths["psepheda"], output)
            save_output()
        except Exception as error:
            output.append("psepheda: αποτυχία ελέγχου contributors")
            output.append(f"  Ακριβές σφάλμα: {type(error).__name__}: {error}")
            output.append("  Traceback:")
            output.extend(
                f"    {line}"
                for line in traceback.format_exc().rstrip().splitlines()
            )
            save_output()

    for dataset in ("libduth", "elocus"):
        if dataset not in paths:
            continue
        print(f"Checking contributors for {dataset}...", flush=True)
        try:
            inspect_contributors_column(dataset, paths[dataset], output)
            save_output()
        except Exception as error:
            output.append(f"{dataset}: αποτυχία ελέγχου contributors")
            output.append(f"  Ακριβές σφάλμα: {type(error).__name__}: {error}")
            output.append("  Traceback:")
            output.extend(
                f"    {line}"
                for line in traceback.format_exc().rstrip().splitlines()
            )
            save_output()

    output.append("")
    output.append("Μέρος Β — περιεχόμενο με DuckDB")
    save_output()
    for dataset in ("libduth", "elocus"):
        if dataset in paths:
            operation = f"DuckDB query for {dataset}"
            inspect_content(dataset, paths[dataset], output)
            operation = "dataset processing"
            save_output()
        else:
            output.append(f"{dataset}: δεν έγινε ερώτημα λόγω αποτυχίας λήψης")
            save_output()


if __name__ == "__main__":
    main()
