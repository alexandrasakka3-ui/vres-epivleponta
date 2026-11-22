from collections import Counter
import random
import re
import unicodedata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


DATASETS = (
    (
        "libduth",
        "glossAPI/libduth",
        "libduth.parquet",
        ["title", "abstract", "advisor", "publication_date", "institution", "url", "pdf_url"],
    ),
    (
        "elocus",
        "glossAPI/elocus",
        "elocus.parquet",
        ["title", "abstract", "advisor", "committee_members", "date", "collection", "url", "pdf_url"],
    ),
    (
        "psepheda",
        "glossAPI/psepheda",
        "psepheda.parquet",
        ["title", "abstract", "supervisor", "contributors", "issue_date", "department", "handle_url", "pdf_url"],
    ),
)

NON_PERSON_TERMS = (
    "προγραμμα",
    "πανεπιστημιο",
    "τμημα",
    "σχολη",
    "μεταπτυχιακ",
    "διατμηματικ",
    "σπουδ",
    "ινστιτουτο",
    "εργαστηριο",
)
DEPARTMENT_TERMS = ("προγραμμα", "πανεπιστημιο", "τμημα")


def clean_text(value):
    if value is None:
        return ""
    return " ".join(str(value).split())


def split_names(value, separator):
    return [name for part in (value or "").split(separator) if (name := clean_text(part))]


def normalized_name(value):
    decomposed = unicodedata.normalize("NFD", clean_text(value).casefold())
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def clean_committee(supervisor, committee, department):
    supervisor_name = normalized_name(supervisor)
    seen = set()
    cleaned = []
    department_entries = []
    seen_department_entries = set()
    for member in committee:
        name = clean_text(member)
        normalized = normalized_name(name)
        if not normalized:
            continue
        if any(term in normalized for term in NON_PERSON_TERMS):
            if (
                not department
                and any(term in normalized for term in DEPARTMENT_TERMS)
                and normalized not in seen_department_entries
            ):
                department_entries.append(name)
                seen_department_entries.add(normalized)
            continue
        if normalized == supervisor_name or normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(name)
    if not department and department_entries:
        department = "; ".join(department_entries)
    return cleaned, department


def extract_year(value):
    match = re.search(r"\b(\d{4})\b", clean_text(value))
    return int(match.group(1)) if match else None


def elocus_department(collection):
    sections = clean_text(collection).split("--")
    departments = [section for section in sections if "τμήμα" in section.casefold()]
    return clean_text(departments[-1]) if departments else ""


def psepheda_has_department_entry(entries):
    if not entries:
        return False
    last_entry = entries[-1]
    return (
        last_entry.casefold().startswith("πανεπιστήμιο")
        or "τμήμα" in last_entry.casefold()
    )


def psepheda_department(entries, source_department):
    if psepheda_has_department_entry(entries):
        return entries[-1]
    return clean_text(source_department)


def make_record(dataset, row):
    abstract = clean_text(row.get("abstract"))
    if len(abstract) < 50:
        return None

    committee = []
    if dataset == "libduth":
        supervisor = clean_text(row.get("advisor"))
        committee = split_names(row.get("committee_members"), ";")
        department = clean_text(row.get("institution"))
        year = extract_year(row.get("publication_date"))
        url = clean_text(row.get("url"))
    elif dataset == "elocus":
        advisors = split_names(row.get("advisor"), "\n")
        supervisor = advisors[0] if advisors else ""
        committee = advisors[1:]
        committee.extend(split_names(row.get("committee_members"), ";"))
        department = elocus_department(row.get("collection"))
        year = extract_year(row.get("date"))
        url = clean_text(row.get("url"))
    else:
        supervisor = clean_text(row.get("supervisor"))
        contributors = split_names(row.get("contributors"), ";")
        committee = contributors[1:]
        has_department_entry = psepheda_has_department_entry(contributors)
        department = psepheda_department(contributors, row.get("department"))
        year = extract_year(row.get("issue_date"))
        url = clean_text(row.get("handle_url"))
        if has_department_entry:
            committee = contributors[1:-1]

    if not supervisor:
        return None

    committee, department = clean_committee(supervisor, committee, department)

    return {
        "institution": {
            "libduth": "Θράκη",
            "elocus": "Κρήτη",
            "psepheda": "Μακεδονία",
        }[dataset],
        "title": clean_text(row.get("title")),
        "abstract": abstract,
        "year": year,
        "supervisor": supervisor,
        "committee": committee,
        "department": department,
        "url": url,
        "pdf_url": clean_text(row.get("pdf_url")),
    }


def read_dataset(name, repo_id, filename, columns):
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
    )
    parquet_file = pq.ParquetFile(path)
    schema_names = parquet_file.schema_arrow.names
    missing_columns = [column for column in columns if column not in schema_names]
    if missing_columns:
        raise ValueError(f"{repo_id} is missing required columns: {', '.join(missing_columns)}")

    if name == "libduth":
        if "committee_members" in schema_names:
            columns = [*columns, "committee_members"]
        else:
            print("ΠΡΟΣΟΧΗ: Το libduth.parquet δεν έχει committee_members· το committee στη Θράκη θα μείνει κενό.")

    records = []
    for batch in parquet_file.iter_batches(batch_size=512, columns=columns):
        for row in pa.Table.from_batches([batch]).to_pylist():
            record = make_record(name, row)
            if record is not None:
                records.append(record)
    return records


def main():
    records = []
    print("Στήλη ημερομηνίας για elocus: date")
    for dataset in DATASETS:
        records.extend(read_dataset(*dataset))

    schema = pa.schema(
        [
            ("institution", pa.string()),
            ("title", pa.string()),
            ("abstract", pa.string()),
            ("year", pa.int64()),
            ("supervisor", pa.string()),
            ("committee", pa.list_(pa.string())),
            ("department", pa.string()),
            ("url", pa.string()),
            ("pdf_url", pa.string()),
        ]
    )
    table = pa.Table.from_pylist(records, schema=schema)
    output_path = Path(__file__).resolve().parent / "unified.parquet"
    pq.write_table(table, output_path, compression="zstd")

    print("Στατιστικά και παραδείγματα ανά ίδρυμα:")
    for institution in ("Θράκη", "Κρήτη", "Μακεδονία"):
        institution_records = [
            record for record in records if record["institution"] == institution
        ]
        count = len(institution_records)
        committee_count = sum(bool(record["committee"]) for record in institution_records)
        percentage = 100 * committee_count / count if count else 0
        member_counts = {size: 0 for size in ("1", "2", "3", "4+")}
        for record in institution_records:
            size = len(record["committee"])
            if size:
                bucket = str(size) if size < 4 else "4+"
                member_counts[bucket] += 1

        print(
            f"  {institution}: {count:,} γραμμές, μη κενό committee "
            f"{percentage:.2f}% ({committee_count:,}/{count:,})"
        )
        print(
            "    Γραμμές με μέλη committee: "
            + ", ".join(f"{size}: {member_counts[size]:,}" for size in ("1", "2", "3", "4+"))
        )
        print("    25 συχνότερα στοιχεία committee:")
        for member, frequency in Counter(
            member
            for record in institution_records
            for member in record["committee"]
        ).most_common(25):
            print(f"      {member!r}: {frequency:,}")
        print("    5 τυχαία παραδείγματα (supervisor + committee):")
        for record in random.sample(institution_records, min(5, count)):
            print(
                f"      supervisor={record['supervisor']!r}; "
                f"committee={record['committee']!r}"
            )

    file_size = output_path.stat().st_size
    print(f"Μέγεθος αρχείου: {file_size:,} bytes ({file_size / 1_000_000:.2f} MB)")


if __name__ == "__main__":
    main()
