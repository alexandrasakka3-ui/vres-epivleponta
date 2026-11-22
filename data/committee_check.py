from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


DATASETS = (
    ("glossAPI/libduth", "libduth.parquet", "advisor", None),
    ("glossAPI/elocus", "elocus.parquet", "advisor", None),
    ("glossAPI/psepheda", "psepheda.parquet", "supervisor", "contributors"),
)

CONTENT_TERMS = ("επιτροπ", "Τριμελ", "Εξεταστ", "Εξεταστικ")


def text_value(value):
    if value is None:
        return ""
    return str(value)


def separator_for(value):
    if ";" in value:
        return "ερωτηματικό (;)"
    if "||| " in value or " ||| " in value:
        return "τρεις κάθετες (|||)"
    if "|||":
        if " ||| " in value:
            return "τρεις κάθετες (|||)"
    if "\n" in value:
        return "αλλαγή γραμμής"
    if "|" in value:
        return "κάθετη (|)"
    if "&" in value:
        return "περιττό και (&)"
    if " και " in value.casefold():
        return "λέξη «και»"
    if "," in value and value.count(",") >= 2:
        return "κόμμα (,)"
    if "/" in value:
        return "κάθετος (/)"
    return "άλλος/άγνωστος διαχωριστής"


def has_multiple_names(value):
    text = text_value(value).strip()
    if not text:
        return False
    if ";" in text:
        return any(part.strip() for part in text.split(";")) and len(
            [part for part in text.split(";") if part.strip()]
        ) > 1
    for separator in ("|||", "|", "\n", "&"):
        if separator in text:
            return len([part for part in text.split(separator) if part.strip()]) > 1
    if " και " in text.casefold():
        return len([part for part in text.casefold().split(" και ") if part.strip()]) > 1
    if text.count(",") >= 2:
        return True
    return False


def percentage(count, total):
    if total == 0:
        return "0.00%"
    return f"{100 * count / total:.2f}%"


def inspect_name_column(parquet_file, column, output):
    if column not in parquet_file.schema_arrow.names:
        output.append(f"{column}: η στήλη δεν υπάρχει")
        return

    first_values = []
    multiple_count = 0
    total_rows = 0
    separators = set()

    for batch in parquet_file.iter_batches(
        batch_size=64,
        columns=[column],
        use_threads=True,
    ):
        table = pa.Table.from_batches([batch])
        values = table.column(column).to_pylist()
        for value in values:
            text = text_value(value)
            total_rows += 1
            if len(first_values) < 10:
                first_values.append(text)
            if has_multiple_names(text):
                multiple_count += 1
                separators.add(separator_for(text))

    output.append(f"{column}:")
    for index, value in enumerate(first_values, start=1):
        output.append(f"  Row {index}: {value}")
    output.append(
        f"  Ποσοστό γραμμών με >1 ονόματα: "
        f"{percentage(multiple_count, total_rows)} "
        f"({multiple_count}/{total_rows})"
    )
    if separators:
        output.append(f"  Διαχωριστικό: {', '.join(sorted(separators))}")
    else:
        output.append("  Διαχωριστικό: δεν εντοπίστηκε")


def inspect_content(parquet_file, output):
    matches = 0
    total_texts = 0
    examples = []

    content_index = parquet_file.schema_arrow.names.index("content")
    uncompressed_size = sum(
        parquet_file.metadata.row_group(row_group).column(content_index).total_uncompressed_size
        for row_group in range(parquet_file.metadata.num_row_groups)
    )
    if uncompressed_size > 1_000_000_000:
        output.append("content:")
        output.append(
            "  Δεν ήταν δυνατή η ανάγνωση: η στήλη content είναι πάνω από "
            f"{uncompressed_size / 1_000_000_000:.2f} GB αποσυμπιεσμένα "
            "και υπερβαίνει το διαθέσιμο όριο μνήμης."
        )
        output.append("  Δεν υπολογίστηκε ποσοστό χωρίς να διαβαστούν τα κείμενα.")
        return

    for batch in parquet_file.iter_batches(
        batch_size=300,
        columns=["content"],
        use_threads=True,
    ):
        table = pa.Table.from_batches([batch])
        values = table.column("content").to_pylist()
        for value in values:
            if total_texts >= 300:
                break
            total_texts += 1
            text = text_value(value)[:6000]
            folded = text.casefold()
            first_index = None
            for term in CONTENT_TERMS:
                index = folded.find(term.casefold())
                if index != -1 and (first_index is None or index < first_index):
                    first_index = index

            if first_index is not None:
                matches += 1
                if len(examples) < 3:
                    start = max(0, first_index - 200)
                    excerpt = text[start:start + 400]
                    if len(excerpt) < 400:
                        start = max(0, len(text) - 400)
                        excerpt = text[start:start + 400]
                    examples.append((total_texts, excerpt))

    output.append("content:")
    output.append(
        f"  Ποσοστό των πρώτων 300 κειμένων που αναφέρουν επιτροπή: "
        f"{percentage(matches, total_texts)} ({matches}/{total_texts})"
    )
    output.append("  3 παραδείγματα γύρω από την πρώτη εμφάνιση:")
    if not examples:
        output.append("  Κανένα παράδειγμα δεν βρέθηκε.")
    else:
        for row_number, excerpt in examples:
            output.append(f"  Row {row_number}: {excerpt}")


def inspect_dataset(dataset_name, filename, supervisor_column, contributor_column, output):
    local_path = hf_hub_download(
        repo_id=dataset_name,
        filename=filename,
        repo_type="dataset",
    )
    parquet_file = pq.ParquetFile(local_path)

    output.append(dataset_name)
    inspect_name_column(parquet_file, supervisor_column, output)
    if contributor_column:
        inspect_name_column(parquet_file, contributor_column, output)

    if "content" not in parquet_file.schema_arrow.names:
        output.append("content: η στήλη δεν υπάρχει")
    else:
        inspect_content(parquet_file, output)
    output.append("")


def main():
    output = []
    for dataset_name, filename, supervisor_column, contributor_column in DATASETS:
        inspect_dataset(
            dataset_name,
            filename,
            supervisor_column,
            contributor_column,
            output,
        )

    output_path = Path(__file__).resolve().parent.parent / "committee_check.txt"
    output_path.write_text("\n".join(output), encoding="utf-8")


if __name__ == "__main__":
    main()
