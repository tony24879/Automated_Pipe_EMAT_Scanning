"""Create an error CSV by dividing column 10 values from two raw scan CSV files.

The output CSV is a copy of the first input CSV, but column 10 (1-based) is
replaced row-by-row with:

    first_csv_col10 / second_csv_col10

Output filename format:
    <first_stem>_<second_stem>_errors.csv
"""

from __future__ import annotations

import argparse
import csv
from itertools import zip_longest
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"
TARGET_COLUMN_ONE_BASED = 10
TARGET_COLUMN_INDEX = TARGET_COLUMN_ONE_BASED - 1


def blank_columns_after_target(row: list[str]) -> list[str]:
    output_row = list(row)
    for col_index in range(TARGET_COLUMN_INDEX + 1, len(output_row)):
        output_row[col_index] = ""
    return output_row


def resolve_input_path(raw_file: str) -> Path:
    candidate = Path(raw_file)
    if candidate.is_absolute():
        return candidate
    return RAW_DIR / candidate


def build_output_path(first_csv: Path, second_csv: Path, output_folder: Path | None = None) -> Path:
    folder = RAW_DIR if output_folder is None else output_folder
    return folder / f"{first_csv.stem}_{second_csv.stem}_errors.csv"


def parse_numeric(cell: str, row_number: int, file_path: Path) -> float:
    text = cell.strip()
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric value in column {TARGET_COLUMN_ONE_BASED} at row {row_number} "
            f"in {file_path.name}: {cell!r}"
        ) from exc


def process_files(first_csv: Path, second_csv: Path, output_folder: Path | None = None) -> Path:
    output_csv = build_output_path(first_csv, second_csv, output_folder=output_folder)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with first_csv.open("r", newline="", encoding="utf-8") as first_file, second_csv.open(
        "r", newline="", encoding="utf-8"
    ) as second_file, output_csv.open("w", newline="", encoding="utf-8") as output_file:
        first_reader = csv.reader(first_file)
        second_reader = csv.reader(second_file)
        writer = csv.writer(output_file)

        rows_written = 0

        # zip_longest enforces strict row-count parity so scans stay aligned.
        for row_number, (first_row, second_row) in enumerate(
            zip_longest(first_reader, second_reader), start=1
        ):
            if first_row is None or second_row is None:
                raise ValueError(
                    "Input CSV files have different numbers of rows. "
                    f"Mismatch found at row {row_number}."
                )

            if len(first_row) <= TARGET_COLUMN_INDEX or len(second_row) <= TARGET_COLUMN_INDEX:
                raise ValueError(
                    f"Row {row_number} does not contain column {TARGET_COLUMN_ONE_BASED} in one "
                    "or both files."
                )

            # Preserve possible header rows by copying non-numeric column-10 entries as-is.
            try:
                first_value = parse_numeric(first_row[TARGET_COLUMN_INDEX], row_number, first_csv)
                second_value = parse_numeric(second_row[TARGET_COLUMN_INDEX], row_number, second_csv)
            except ValueError:
                if row_number == 1:
                    writer.writerow(blank_columns_after_target(first_row))
                    rows_written += 1
                    continue
                raise

            if second_value == 0:
                raise ZeroDivisionError(
                    f"Division by zero in column {TARGET_COLUMN_ONE_BASED} at row {row_number} "
                    f"of {second_csv.name}."
                )

            updated_row = blank_columns_after_target(first_row)
            updated_row[TARGET_COLUMN_INDEX] = str(first_value / second_value)
            writer.writerow(updated_row)
            rows_written += 1

    print(f"Processed {rows_written} rows.")
    print(f"Output written to: {output_csv}")
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Divide column 10 values in one raw CSV by column 10 values in another and "
            "write a new raw CSV."
        )
    )
    parser.add_argument(
        "first_csv",
        help="First input CSV filename in data/raw (or absolute path).",
    )
    parser.add_argument(
        "second_csv",
        help="Second input CSV filename in data/raw (or absolute path).",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=None,
        help="Optional output folder for the generated *_errors CSV.",
    )
    args = parser.parse_args()

    first_csv = resolve_input_path(args.first_csv)
    second_csv = resolve_input_path(args.second_csv)

    if not first_csv.exists():
        raise FileNotFoundError(f"First input CSV not found: {first_csv}")
    if not second_csv.exists():
        raise FileNotFoundError(f"Second input CSV not found: {second_csv}")

    process_files(first_csv, second_csv, output_folder=args.output_folder)


if __name__ == "__main__":
    main()