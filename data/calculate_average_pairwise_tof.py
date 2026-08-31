"""Calculate averaged pairwise TOF values from waveform peaks in scan CSV files.

Input format expected from sync logger:
- Columns 1-9: pose/scan metadata
- Column 10: existing TOF (replaced in-place)
- Columns 11+: waveform signal samples

For each row:
- Parse signal samples from column 11 onward.
- Detect peaks using the same method/constants as the row-peak workflow.
- Keep the first N detected peaks (`--num-peaks`).
- Compute TOF for each adjacent pair: (peak1, peak2), (peak2, peak3), ...
- Average the pairwise TOF values.
- Overwrite column 10 in the original CSV with the new calculated TOF.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

RAW_DIR = Path(__file__).resolve().parent / "raw"

# Keep these aligned with the row peak-detection workflow in plot_row_peaks.py.
SKIP_SAMPLES = 200
MIN_PEAK_DISTANCE = 250
MIN_PROMINENCE = 100
TOF_SCALE_SECONDS = 20e-9


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


def find_first_peak_in_branch(signal_values: list[float]) -> tuple[int, str, float] | None:
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
        (SKIP_SAMPLES + int(idx), "normal", float(signal_values[SKIP_SAMPLES + int(idx)]))
        for idx in pos_peak_indices
    ]
    negative_peaks = [
        (SKIP_SAMPLES + int(idx), "flipped", float(signal_values[SKIP_SAMPLES + int(idx)]))
        for idx in neg_peak_indices
    ]

    first_candidates: list[tuple[int, str, float]] = []
    if positive_peaks:
        first_candidates.append(positive_peaks[0])
    if negative_peaks:
        first_candidates.append(negative_peaks[0])
    if not first_candidates:
        return None

    return max(first_candidates, key=lambda item: abs(item[2]))


def find_peak_sequence(signal_values: list[float], num_peaks: int) -> list[int]:
    """Return the first peaks from the branch selected by the strongest first peak."""
    if num_peaks < 2:
        raise ValueError("num_peaks must be at least 2.")

    if len(signal_values) <= SKIP_SAMPLES + 1:
        return []

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
        (SKIP_SAMPLES + int(idx), "normal", float(signal_values[SKIP_SAMPLES + int(idx)]))
        for idx in pos_peak_indices
    ]
    negative_peaks = [
        (SKIP_SAMPLES + int(idx), "flipped", float(signal_values[SKIP_SAMPLES + int(idx)]))
        for idx in neg_peak_indices
    ]

    first_candidates: list[tuple[int, str, float]] = []
    if positive_peaks:
        first_candidates.append(positive_peaks[0])
    if negative_peaks:
        first_candidates.append(negative_peaks[0])
    if not first_candidates:
        return []

    first_peak = max(first_candidates, key=lambda item: abs(item[2]))
    same_polarity_peaks = positive_peaks if first_peak[1] == "normal" else negative_peaks
    selected = [first_peak, *same_polarity_peaks[1:num_peaks]]

    return [peak[0] for peak in selected]


def compute_avg_pairwise_tof(signal_values: list[float], num_peaks: int) -> str:
    """Return averaged adjacent-pair TOF in seconds, or empty string if unavailable."""
    if num_peaks < 2:
        raise ValueError("num_peaks must be at least 2.")

    peak_indices = find_peak_sequence(signal_values, num_peaks=num_peaks)
    if len(peak_indices) < 2:
        return ""

    interpolated_peak_indices = [
        interpolate_peak_zero_crossing(signal_values, peak_index)[0]
        for peak_index in peak_indices
    ]

    pairwise_tofs: list[float] = []
    for i in range(len(interpolated_peak_indices) - 1):
        sample_difference = interpolated_peak_indices[i + 1] - interpolated_peak_indices[i]
        pairwise_tofs.append(sample_difference * TOF_SCALE_SECONDS)

    if not pairwise_tofs:
        return ""

    avg_tof = sum(pairwise_tofs) / len(pairwise_tofs)
    return str(avg_tof)


def process_file(input_csv: Path, num_peaks: int, output_file: Path | None = None) -> Path:
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

        try:
            signal_values = parse_signal_columns(working_row)
        except ValueError as exc:
            raise ValueError(f"Failed parsing signal values at row {row_idx}: {exc}") from exc

        avg_tof = compute_avg_pairwise_tof(signal_values, num_peaks=num_peaks)
        working_row[9] = avg_tof
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
        description=(
            "Compute averaged adjacent-pair TOF from first N detected peaks in CSV signal columns and overwrite column 10."
        )
    )
    parser.add_argument(
        "input_csv",
        help="Input CSV filename in data/raw (or an absolute path).",
    )
    parser.add_argument(
        "--num-peaks",
        type=int,
        required=True,
        help="Number of detected peaks to use. Must be >= 2.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional explicit output CSV path. Defaults to overwriting the input file in place.",
    )
    args = parser.parse_args()

    if args.num_peaks < 2:
        raise ValueError("--num-peaks must be at least 2.")

    input_csv = resolve_input_path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    process_file(input_csv, num_peaks=args.num_peaks, output_file=args.output_file)


if __name__ == "__main__":
    main()
