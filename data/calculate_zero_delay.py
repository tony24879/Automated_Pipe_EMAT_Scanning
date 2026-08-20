"""Calculate the average zero-delay value from interpolated peak positions.

This mirrors the workflow used in recalculate_tof_column10.py up to the point where the
first two signal peaks are found and interpolated. For each valid row, it computes:

    peak_1_interpolated_x - (peak_2_interpolated_x - peak_1_interpolated_x)

and then outputs the average of those values across all valid rows.
Rows where fewer than two valid peaks are found are ignored.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from recalculate_tof_column10 import (
    find_two_signal_peaks,
    interpolate_peak_zero_crossing,
    parse_signal_columns,
    resolve_input_csv_path,
)


def compute_zero_delay_for_row(signal_values: list[float]) -> float | None:
    """Compute the zero-delay value for a single signal row, or return None if invalid."""
    peaks = find_two_signal_peaks(signal_values)
    if peaks is None:
        return None

    signal = np.asarray(signal_values, dtype=float)
    first_peak_idx, second_peak_idx = peaks

    first_interp_x, _ = interpolate_peak_zero_crossing(signal, first_peak_idx)
    second_interp_x, _ = interpolate_peak_zero_crossing(signal, second_peak_idx)

    zero_delay_value = first_interp_x - (second_interp_x - first_interp_x)
    return float(zero_delay_value)


def calculate_average_zero_delay(input_csv: Path) -> float:
    """Process the input CSV and return the average zero-delay value across valid rows."""
    with input_csv.open("r", newline="", encoding="utf-8") as source_file:
        reader = csv.reader(source_file)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Input file is empty: {input_csv}")

    values: list[float] = []
    for row in rows[1:]:
        if not row:
            continue

        signal_values = parse_signal_columns(row)
        if not signal_values:
            continue

        result = compute_zero_delay_for_row(signal_values)
        if result is None:
            continue

        values.append(result)

    if not values:
        raise ValueError("No valid rows were found with two detectable peaks.")

    return float(np.mean(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average the zero-delay metric derived from the first two interpolated peak positions."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Input CSV file path (absolute or relative; relative also checks data/raw).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = resolve_input_csv_path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    average_zero_delay = calculate_average_zero_delay(input_csv)
    print(f"Average zero delay: {average_zero_delay}")


if __name__ == "__main__":
    main()
