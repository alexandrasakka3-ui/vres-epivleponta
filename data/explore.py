from itertools import islice
from collections.abc import Mapping
import sys

from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError


DATASETS = (
    "glossAPI/libduth",
    "glossAPI/elocus",
    "glossAPI/psepheda",
)


def show_dataset(dataset_name):
    print(f"\n{'=' * 80}\nDataset: {dataset_name}")
    try:
        loaded = load_dataset(dataset_name, streaming=True)
    except DatasetNotFoundError as error:
        print(f"Could not load dataset: {error}")
        return False

    splits = loaded.items() if isinstance(loaded, Mapping) else (("data", loaded),)

    for split_name, dataset in splits:
        print(f"\nSplit: {split_name}")
        columns = dataset.column_names
        print(f"Columns: {columns}")

        for row_number, row in enumerate(islice(dataset, 3), start=1):
            print(f"\nRow {row_number}:")
            for column in columns:
                value = str(row.get(column, ""))[:200]
                print(f"  {column}: {value}")
    return True


def main():
    failed = [
        dataset_name
        for dataset_name in DATASETS
        if not show_dataset(dataset_name)
    ]
    if failed:
        print(
            "\nSome datasets could not be explored. Request access on the "
            "Hugging Face dataset pages, then authenticate before rerunning.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
