from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


DATASETS = (
    ("glossAPI/libduth", "libduth.parquet"),
    ("glossAPI/elocus", "elocus.parquet"),
    ("glossAPI/psepheda", "psepheda.parquet"),
)


def truncate(value, length=120):
    return str(value)[:length]


def inspect_dataset(dataset_name, filename, output):
    local_path = hf_hub_download(
        repo_id=dataset_name,
        filename=filename,
        repo_type="dataset",
    )
    parquet_file = pq.ParquetFile(local_path)

    output.append(dataset_name)
    schema = parquet_file.schema_arrow
    for field in schema:
        output.append(f"{field.name}: {field.type}")

    columns = [field.name for field in schema if field.name != "content"]
    output.append("First 3 rows:")
    rows_written = 0

    for batch in parquet_file.iter_batches(
        batch_size=3,
        columns=columns,
        use_threads=True,
    ):
        batch_table = pa.Table.from_batches([batch])
        for row_index in range(batch.num_rows):
            if rows_written >= 3:
                break
            rows_written += 1
            output.append(f"Row {rows_written}:")
            for column in columns:
                value = batch_table.column(column)[row_index].as_py()
                output.append(f"{column}: {truncate(value)}")
        if rows_written >= 3:
            break

    output.append("")


def main():
    output = []
    for dataset_name, filename in DATASETS:
        inspect_dataset(dataset_name, filename, output)

    output_path = Path(__file__).resolve().parent.parent / "columns.txt"
    output_path.write_text("\n".join(output), encoding="utf-8")


if __name__ == "__main__":
    main()
