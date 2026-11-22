import sys
import traceback

import pyarrow.parquet as parquet
from huggingface_hub import hf_hub_download, list_repo_files


DATASETS = (
    "glossAPI/libduth",
    "glossAPI/elocus",
    "glossAPI/psepheda",
)


def inspect_dataset(dataset_name):
    print(f"\n{'=' * 80}\nDataset: {dataset_name}", flush=True)
    parquet_files = [
        filename
        for filename in list_repo_files(dataset_name, repo_type="dataset")
        if filename.lower().endswith(".parquet")
    ]
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {dataset_name}")

    filename = parquet_files[0]
    print(f"Parquet file: {filename}", flush=True)
    local_path = hf_hub_download(
        repo_id=dataset_name,
        filename=filename,
        repo_type="dataset",
    )
    parquet_file = parquet.ParquetFile(local_path)
    print(f"Rows: {parquet_file.metadata.num_rows}", flush=True)
    print(f"Columns: {parquet_file.schema.names}", flush=True)


def main():
    failures = 0
    for dataset_name in DATASETS:
        try:
            inspect_dataset(dataset_name)
        except Exception:
            failures += 1
            traceback.print_exc()

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
