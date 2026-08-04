"""Fix column-10 ordering for scans captured in reversed theta/axis directions.

This script reads one CSV, takes column 10 values (1-based) from data rows,
splits them into groups of 40, then reverses only the order of those groups.
Values inside each group are preserved in their original order.

It writes a new CSV with every cell unchanged except column 10, which is
replaced by the transformed sequence.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"
TARGET_COLUMN_ONE_BASED = 10
TARGET_COLUMN_INDEX = TARGET_COLUMN_ONE_BASED - 1
DEFAULT_GROUP_SIZE = 40


def resolve_input_path(input_csv: str) -> Path:
    candidate = Path(input_csv)
    if candidate.is_absolute():
        return candidate
    return RAW_DIR / candidate


def build_output_path(input_csv: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return input_csv.with_name(f"{input_csv.stem}_fixed_col10{input_csv.suffix}")


def is_numeric(text: str) -> bool:
    try:
        float(text.strip())
        return True
    except ValueError:
        return False


def transform_values(values: list[str], group_size: int) -> list[str]:
    if group_size <= 0:
        raise ValueError("Group size must be > 0.")

    if len(values) % group_size != 0:
        raise ValueError(
            f"Column 10 contains {len(values)} data values, which is not divisible by group size {group_size}."
        )

    groups = [values[i : i + group_size] for i in range(0, len(values), group_size)]
    # Reverse only group order (scan block order), not the per-group sample order.
    reversed_group_order = list(reversed(groups))
    return [value for group in reversed_group_order for value in group]


def process_file(input_csv: Path, output_csv: Path, group_size: int) -> Path:
    with input_csv.open("r", newline="", encoding="utf-8") as source_file:
        rows = list(csv.reader(source_file))

    if not rows:
        raise ValueError(f"Input CSV is empty: {input_csv}")

    if len(rows[0]) <= TARGET_COLUMN_INDEX:
        raise ValueError(
            f"Input CSV does not contain column {TARGET_COLUMN_ONE_BASED}: {input_csv}"
        )

    has_header = not is_numeric(rows[0][TARGET_COLUMN_INDEX])
    header_rows = rows[:1] if has_header else []
    data_rows = rows[1:] if has_header else rows

    if not data_rows:
        raise ValueError("Input CSV has no data rows to transform.")

    for row_number, row in enumerate(data_rows, start=2 if has_header else 1):
        if len(row) <= TARGET_COLUMN_INDEX:
            raise ValueError(
                f"Row {row_number} does not contain column {TARGET_COLUMN_ONE_BASED}."
            )

    original_values = [row[TARGET_COLUMN_INDEX] for row in data_rows]
    transformed_values = transform_values(original_values, group_size=group_size)

    for row, new_value in zip(data_rows, transformed_values):
        row[TARGET_COLUMN_INDEX] = new_value

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as dest_file:
        writer = csv.writer(dest_file)
        writer.writerows([*header_rows, *data_rows])

    print(f"Processed {len(data_rows)} data rows.")
    print(f"Output written to: {output_csv}")
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fix flipped scan ordering by reversing only the group order of column-10 values."
        )
    )
    parser.add_argument(
        "input_csv",
        help="Input CSV filename in data/raw (or absolute path).",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=DEFAULT_GROUP_SIZE,
        help=f"Number of rows per group (default: {DEFAULT_GROUP_SIZE}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to <input>_fixed_col10.csv",
    )
    args = parser.parse_args()

    input_csv = resolve_input_path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_csv = build_output_path(input_csv, args.output)
    process_file(input_csv, output_csv=output_csv, group_size=args.group_size)


if __name__ == "__main__":
    main()
