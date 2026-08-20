"""Recalculate the TOF value in-column 10 using interpolated peak indices.

Behavior:
- Reads an input CSV from an absolute path or from data/raw when relative.
- For each data row, parses waveform samples from column 11 onward (1-based).
- Detects the first two peaks using the same scipy peak finder as the row-plot workflow.
- Uses zero-crossing interpolated peak indices to compute the time difference.
- Writes only the recalculated TOF to column 10 in the original CSV.
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
SIGNAL_START_COL_1_BASED = 11


def resolve_input_csv_path(input_path: Path) -> Path:
    """Resolve input path using existing repo conventions."""
    candidates: list[Path] = [input_path]

    if input_path.suffix == "":
        candidates.append(input_path.with_suffix(".csv"))

    if not input_path.is_absolute():
        candidates.append(RAW_DIR / input_path.name)
        if input_path.suffix == "":
            candidates.append(RAW_DIR / f"{input_path.name}.csv")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def parse_signal_columns(row: list[str], start_col_1_based: int = SIGNAL_START_COL_1_BASED) -> list[float]:
    """Parse waveform values from start column onward, skipping empty cells."""
    start_idx = start_col_1_based - 1
    if len(row) <= start_idx:
        return []

    values: list[float] = []
    for cell in row[start_idx:]:
        text = cell.strip()
        if text == "":
            continue
        values.append(float(text))
    return values


def interpolate_peak_zero_crossing(signal: np.ndarray, peak_idx: int) -> tuple[float, float]:
    """Estimate the sub-sample peak location using a local quadratic fit."""
    if peak_idx <= 0 or peak_idx >= signal.size - 1:
        return float(peak_idx), float(signal[peak_idx])

    y_minus = float(signal[peak_idx - 1])
    y0 = float(signal[peak_idx])
    y_plus = float(signal[peak_idx + 1])
    denominator = y_minus - 2.0 * y0 + y_plus

    if abs(denominator) < 1e-12:
        return float(peak_idx), y0

    delta = 0.5 * (y_minus - y_plus) / denominator
    interpolated_x = float(peak_idx) + delta
    interpolated_y = (
        y0
        + 0.5 * (y_plus - y_minus) * delta
        + 0.5 * (y_minus - 2.0 * y0 + y_plus) * delta**2
    )
    return interpolated_x, float(interpolated_y)


def find_two_signal_peaks(signal_values: list[float]) -> tuple[int, int] | None:
    """Return the two strongest adjacent same-polarity peaks using the row-plot workflow."""
    if len(signal_values) <= SKIP_SAMPLES + 1:
        return None

    signal = np.asarray(signal_values, dtype=float)
    post_noise = signal[SKIP_SAMPLES:]
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
        (SKIP_SAMPLES + int(idx), "normal", float(signal[SKIP_SAMPLES + int(idx)]))
        for idx in pos_peak_indices
    ]
    negative_peaks = [
        (SKIP_SAMPLES + int(idx), "flipped", float(signal[SKIP_SAMPLES + int(idx)]))
        for idx in neg_peak_indices
    ]

    first_candidates = []
    if positive_peaks:
        first_candidates.append(positive_peaks[0])
    if negative_peaks:
        first_candidates.append(negative_peaks[0])
    if not first_candidates:
        return None

    first_peak_index, first_source, _ = max(first_candidates, key=lambda item: abs(item[2]))
    same_polarity_peaks = positive_peaks if first_source == "normal" else negative_peaks
    if len(same_polarity_peaks) < 2:
        return None

    second_peak = max(
        same_polarity_peaks[1:],
        key=lambda item: abs(item[2]),
    )
    return first_peak_index, second_peak[0]


def compute_recalculated_tof(signal_values: list[float]) -> str:
    """Compute the interpolated TOF between the first two valid peaks and return it as a string."""
    peaks = find_two_signal_peaks(signal_values)
    if peaks is None:
        return ""

    signal = np.asarray(signal_values, dtype=float)
    first_peak_idx, second_peak_idx = peaks
    first_interp_idx, _ = interpolate_peak_zero_crossing(signal, first_peak_idx)
    second_interp_idx, _ = interpolate_peak_zero_crossing(signal, second_peak_idx)
    return str((second_interp_idx - first_interp_idx) * TOF_SCALE_SECONDS)


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

        signal_values = parse_signal_columns(working_row)
        tof_value = compute_recalculated_tof(signal_values)
        working_row[9] = tof_value
        updated_rows.append(working_row)
        rows_written += 1

    with dest_csv.open("w", newline="", encoding="utf-8") as dest_file:
        writer = csv.writer(dest_file)
        writer.writerows(updated_rows)

    print(f"Processed {rows_written} data rows.")
    print(f"Updated in place: {dest_csv}")
    return dest_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute the TOF in column 10 using the first two interpolated signal peaks."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Input CSV file path (absolute or relative; relative also checks data/raw).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to overwriting the input file in place.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_csv = resolve_input_csv_path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    process_file(input_csv=input_csv, output_file=args.output_file)


if __name__ == "__main__":
    main()
