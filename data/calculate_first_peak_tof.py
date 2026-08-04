"""Calculate first-peak time of flight from raw scan CSV files.

For each data row:
- Read columns 310 to 390 inclusive (1-based).
- Find the maximum value in that range.
- Find the column where that maximum occurs, subtract 10, and multiply by 20e-9
    to convert it to time of flight in seconds.
- Write a new CSV in data/processed containing columns 1-9 from the input and the
  calculated time of flight in column 10.

The output filename is the input filename with "_first_peak_tof" appended.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent / "processed"

# Signal-search bounds are 1-based CSV column indices.
# Keep these as constants so window retuning is explicit and centralized.
FIRST_SIGNAL_COL = 300 #310
LAST_SIGNAL_COL = 400 #390
TOF_SCALE_SECONDS = 20e-9


def to_zero_based(column_number: int) -> int:
    if column_number < 1:
        raise ValueError("Column numbers must be 1-based and >= 1.")
    return column_number - 1


def resolve_input_path(raw_file: str) -> Path:
    candidate = Path(raw_file)
    if candidate.is_absolute():
        return candidate
    return RAW_DIR / candidate


def build_output_path(input_csv: Path, output_folder: Path | None = None) -> Path:
    folder = PROCESSED_DIR if output_folder is None else output_folder
    return folder / f"{input_csv.stem}_first_peak_tof{input_csv.suffix}"


def compute_time_of_flight(row: list[str]) -> str:
    """Return first-peak TOF in seconds, or empty string when not computable."""
    start_idx = to_zero_based(FIRST_SIGNAL_COL)
    end_idx = to_zero_based(LAST_SIGNAL_COL)

    if len(row) <= end_idx:
        return ""

    peak_value = None
    peak_column = None
    for column_number, cell in enumerate(row[start_idx : end_idx + 1], start=FIRST_SIGNAL_COL):
        text = cell.strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError:
            continue

        if peak_value is None or value > peak_value:
            peak_value = value
            peak_column = column_number

    if peak_value is None or peak_column is None:
        return ""

    # Conversion pipeline:
    # 1) Convert detected peak column index to sample time.
    # 2) Apply a fixed calibration offset used by downstream tooling.
    return str(((peak_column - 10) * TOF_SCALE_SECONDS) - 0.00000085)


def process_file(input_csv: Path, output_folder: Path | None = None) -> Path:
    output_csv = build_output_path(input_csv, output_folder=output_folder)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with input_csv.open("r", newline="", encoding="utf-8") as source_file, output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as dest_file:
        reader = csv.reader(source_file)
        writer = csv.writer(dest_file)

        header = next(reader, None)
        if header is None:
            raise ValueError(f"Input file is empty: {input_csv}")

        if len(header) < 9:
            raise ValueError(
                f"Input file must contain at least 9 columns before the signal data: {input_csv}"
            )

        writer.writerow([*header[:9], "Time of Flight (s)"])

        rows_written = 0
        for row_idx, row in enumerate(reader, start=2):
            if len(row) < 9:
                continue

            tof = compute_time_of_flight(row)
            writer.writerow([*row[:9], tof])
            rows_written += 1

    print(f"Processed {rows_written} data rows.")
    print(f"Output written to: {output_csv}")
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate first-peak time of flight from raw scan CSV files."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="sync_scan_20260709_112954.csv",
        help="Input CSV file name in data/raw (or an absolute path).",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=None,
        help="Optional output folder for the *_first_peak_tof CSV.",
    )
    args = parser.parse_args()

    input_csv = resolve_input_path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    process_file(input_csv, output_folder=args.output_folder)


if __name__ == "__main__":
    main()