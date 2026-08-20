"""Calculate a single first-peak TOF from raw scan CSV files.

For each data row:
- Parse signal samples from column 11 onward.
- Detect the first peak using the same scipy peak finder as the pairwise TOF logic.
- Interpolate the peak index using a local quadratic fit.
- Convert the interpolated sample index to TOF using the same calibration offset convention.
- Overwrite column 10 in the original CSV with the calculated TOF value.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

RAW_DIR = Path(__file__).resolve().parent / "raw"
SKIP_SAMPLES = 200
MIN_PEAK_DISTANCE = 250
MIN_PROMINENCE = 30
TOF_SCALE_SECONDS = 20e-9
PEAK_SAMPLE_OFFSET = 34
SIGNAL_START_COL_1_BASED = 11


def resolve_input_path(raw_file: str) -> Path:
    candidate = Path(raw_file)
    if candidate.is_absolute():
        return candidate
    return RAW_DIR / candidate


def parse_signal_columns(row: list[str]) -> list[float]:
    """Parse signal values from column 11 onward (1-based indexing)."""
    values: list[float] = []
    for cell in row[10:]:
        text = cell.strip()
        if text == "":
            continue
        values.append(float(text))
    return values


def interpolate_peak_zero_crossing(signal_values: list[float], peak_idx: int) -> tuple[float, float]:
    """Estimate the sub-sample peak location using a local quadratic fit."""
    signal = np.asarray(signal_values, dtype=float)
    if peak_idx <= 0 or peak_idx >= signal.size - 1:
        return float(peak_idx), float(signal[peak_idx])

    y_minus = float(signal[peak_idx - 1])
    y0 = float(signal[peak_idx])
    y_plus = float(signal[peak_idx + 1])
    denominator = y_minus - 2.0 * y0 + y_plus

    if abs(denominator) < 1e-12:
        return float(peak_idx), y0

    delta = 0.5 * (y_minus - y_plus) / denominator
    interpolated_index = float(peak_idx) + delta
    interpolated_value = (
        y0
        + 0.5 * (y_plus - y_minus) * delta
        + 0.5 * (y_minus - 2.0 * y0 + y_plus) * delta**2
    )
    return interpolated_index, float(interpolated_value)


def find_first_peak(signal_values: list[float]) -> int | None:
    """Return the strongest first peak from either the signal branch or its flipped branch."""
    if len(signal_values) <= SKIP_SAMPLES + 1:
        return None

    post_noise = np.asarray(signal_values[SKIP_SAMPLES:], dtype=float)
    pos_peak_indices, _ = find_peaks(
        post_noise,
        distance=MIN_PEAK_DISTANCE,
        prominence=MIN_PROMINENCE,
    )
    neg_peak_indices, _ = find_peaks(
        -post_noise,
        distance=MIN_PEAK_DISTANCE,
        prominence=MIN_PROMINENCE,
    )

    positive_peaks = [
        (SKIP_SAMPLES + int(idx), float(signal_values[SKIP_SAMPLES + int(idx)]))
        for idx in pos_peak_indices
    ]
    negative_peaks = [
        (SKIP_SAMPLES + int(idx), float(signal_values[SKIP_SAMPLES + int(idx)]))
        for idx in neg_peak_indices
    ]

    first_candidates = []
    if positive_peaks:
        first_candidates.append(positive_peaks[0])
    if negative_peaks:
        first_candidates.append(negative_peaks[0])
    if not first_candidates:
        return None

    first_peak_index, _ = max(first_candidates, key=lambda item: abs(item[1]))
    return int(first_peak_index)


def compute_time_of_flight(row: list[str]) -> str:
    """Return the first-peak TOF in seconds using the interpolated peak index."""
    signal_values = parse_signal_columns(row)
    if not signal_values:
        return ""

    peak_index = find_first_peak(signal_values)
    if peak_index is None:
        return ""

    interpolated_peak_index, _ = interpolate_peak_zero_crossing(signal_values, peak_index)
    peak_column = SIGNAL_START_COL_1_BASED + interpolated_peak_index
    return str(((peak_column - 10 - PEAK_SAMPLE_OFFSET) * TOF_SCALE_SECONDS))


def process_file(input_csv: Path, output_file: Path | None = None) -> Path:
    dest_csv = input_csv if output_file is None else output_file
    dest_csv.parent.mkdir(parents=True, exist_ok=True)

    with input_csv.open("r", newline="", encoding="utf-8") as source_file:
        reader = csv.reader(source_file)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Input file is empty: {input_csv}")

    header = rows[0]
    updated_rows: list[list[str]] = [header]
    rows_written = 0

    for row_idx, row in enumerate(rows[1:], start=2):
        working_row = list(row)
        if len(working_row) < 10:
            continue
        if len(working_row) <= 9:
            working_row.extend([""] * (10 - len(working_row)))

        tof = compute_time_of_flight(working_row)
        working_row[9] = tof
        updated_rows.append(working_row)
        rows_written += 1

    with dest_csv.open("w", newline="", encoding="utf-8") as dest_file:
        writer = csv.writer(dest_file)
        writer.writerows(updated_rows)

    print(f"Processed {rows_written} data rows.")
    print(f"Updated in place: {dest_csv}")
    return dest_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate first-peak TOF from raw scan CSV files and overwrite column 10."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="sync_scan_20260709_112954.csv",
        help="Input CSV file name in data/raw (or an absolute path).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional explicit output CSV path. Defaults to overwriting the input file in place.",
    )
    args = parser.parse_args()

    input_csv = resolve_input_path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    process_file(input_csv, output_file=args.output_file)


if __name__ == "__main__":
    main()