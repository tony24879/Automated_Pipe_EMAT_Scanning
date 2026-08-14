"""Recalculate row-wise TOF from waveform columns and write TOF/peak metadata.

Behavior:
- Reads an input CSV from an absolute path or from data/raw when relative.
- For each data row, parses waveform samples from column 11 onward (1-based).
- Detects the first two raw peaks using the same polarity-aware logic as data/plot_row_peaks.py.
- Computes raw TOF as (second_peak - first_peak) * 20e-9 and writes it to column 10.
- Writes the raw peak indices to columns 11 and 12.
- Repeats the same calculation using zero-crossing interpolated peak indices and writes the
  interpolated TOF to column 13 and the interpolated peak indices to columns 14 and 15.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

RAW_DIR = Path(__file__).resolve().parent / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent / "processed"

# Match data/plot_row_peaks.py peak-finding constants.
SKIP_SAMPLES = 200
MIN_PEAK_DISTANCE = 250
MIN_PROMINENCE = 100

# Match emat/sync_logger.py TOF scaling.
TOF_SCALE_SECONDS = 20e-9
SIGNAL_START_COL_1_BASED = 11
TOF_COL_1_BASED = 10
PEAK1_COL_1_BASED = 11
PEAK2_COL_1_BASED = 12
INTERP_TOF_COL_1_BASED = 13
INTERP_PEAK1_COL_1_BASED = 14
INTERP_PEAK2_COL_1_BASED = 15
INTERP_PEAK3_COL_1_BASED = 16


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


def _is_numeric(text: str) -> bool:
    try:
        float(text.strip())
        return True
    except ValueError:
        return False


def _has_header(rows: list[list[str]], start_col_idx: int, header_mode: str) -> bool:
    if header_mode == "yes":
        return True
    if header_mode == "no":
        return False

    if not rows:
        return False

    first_row = rows[0]
    if len(first_row) <= start_col_idx:
        return False

    return not _is_numeric(first_row[start_col_idx])


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


def find_three_signal_peaks(y: list[float]) -> list[tuple[int, str, float]]:
    """Return the first three raw-signal peaks while keeping each peak on the same polarity branch.

    We compare the first peak found in each branch, then only consider the same branch for the
    second and third peaks. This prevents mixing a positive peak and a negative peak into the
    same trio.
    """
    if len(y) <= SKIP_SAMPLES + 1:
        return []

    signal = np.asarray(y, dtype=float)
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

    if not positive_peaks and not negative_peaks:
        return []

    first_candidates = []
    if positive_peaks:
        first_candidates.append(positive_peaks[0])
    if negative_peaks:
        first_candidates.append(negative_peaks[0])

    first_peak_index, first_source, first_peak_value = max(
        first_candidates,
        key=lambda item: abs(item[2]),
    )

    selected: list[tuple[int, str, float]] = [(first_peak_index, first_source, first_peak_value)]
    same_polarity_peaks = positive_peaks if first_source == "normal" else negative_peaks

    for peak in sorted(same_polarity_peaks[1:], key=lambda item: abs(item[2]), reverse=True)[:2]:
        selected.append((peak[0], peak[1], peak[2]))

    return selected[:3]


def interpolate_peak_zero_crossing(y: np.ndarray, peak_idx: int) -> tuple[float, float]:
    """Estimate the sub-sample peak location using a local quadratic fit."""
    if peak_idx <= 0 or peak_idx >= y.size - 1:
        return float(peak_idx), float(y[peak_idx])

    y_minus = float(y[peak_idx - 1])
    y0 = float(y[peak_idx])
    y_plus = float(y[peak_idx + 1])
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


def compute_tof_and_peaks(signal_values: list[float]) -> tuple[str, str, str, str, str, str, str]:
    """Return raw and interpolated TOF values / indices, or empty strings if unavailable."""
    if len(signal_values) <= SKIP_SAMPLES + 1:
        return "", "", "", "", "", "", ""

    signal = np.asarray(signal_values, dtype=float)
    signal_peaks = find_three_signal_peaks(signal_values)
    if len(signal_peaks) < 2:
        return "", "", "", "", "", "", ""

    first_peak_idx, _, _ = signal_peaks[0]
    second_peak_idx, _, _ = signal_peaks[1]

    raw_tof = str((second_peak_idx - first_peak_idx) * TOF_SCALE_SECONDS)
    first_interp_idx, _ = interpolate_peak_zero_crossing(signal, first_peak_idx)
    second_interp_idx, _ = interpolate_peak_zero_crossing(signal, second_peak_idx)
    interpolated_tof = str((second_interp_idx - first_interp_idx) * TOF_SCALE_SECONDS)

    third_peak_idx = ""
    if len(signal_peaks) >= 3:
        third_peak_idx = str(signal_peaks[2][0])

    third_interp_idx = ""
    if len(signal_peaks) >= 3:
        third_interp_idx, _ = interpolate_peak_zero_crossing(signal, signal_peaks[2][0])

    return (
        raw_tof,
        str(first_peak_idx),
        str(second_peak_idx),
        interpolated_tof,
        str(first_interp_idx),
        str(second_interp_idx),
        str(third_interp_idx),
    )


def build_output_path(input_csv: Path, output_file: Path | None, in_place: bool) -> Path:
    if in_place:
        return input_csv
    if output_file is not None:
        return output_file
    return PROCESSED_DIR / f"{input_csv.stem}_recalc_tof_peaks_col10_15{input_csv.suffix}"


def process_file(input_csv: Path, output_csv: Path, header_mode: str) -> tuple[int, int]:
    """Recalculate TOF for each data row and write raw/interpolated peak data to columns 10-15."""
    with input_csv.open("r", newline="", encoding="utf-8") as source_file:
        rows = list(csv.reader(source_file))

    if not rows:
        raise ValueError(f"Input CSV is empty: {input_csv}")

    start_col_idx = SIGNAL_START_COL_1_BASED - 1
    has_header = _has_header(rows, start_col_idx, header_mode)

    output_rows: list[list[str]] = []
    data_rows = rows
    if has_header:
        header = list(rows[0][:INTERP_PEAK3_COL_1_BASED])
        if len(header) < INTERP_PEAK3_COL_1_BASED:
            header.extend([""] * (INTERP_PEAK3_COL_1_BASED - len(header)))
        header[TOF_COL_1_BASED - 1] = "Time of Flight (s)"
        header[PEAK1_COL_1_BASED - 1] = "Peak 1 Index"
        header[PEAK2_COL_1_BASED - 1] = "Peak 2 Index"
        header[INTERP_TOF_COL_1_BASED - 1] = "Interpolated TOF (s)"
        header[INTERP_PEAK1_COL_1_BASED - 1] = "Interpolated Peak 1 Index"
        header[INTERP_PEAK2_COL_1_BASED - 1] = "Interpolated Peak 2 Index"
        header[INTERP_PEAK3_COL_1_BASED - 1] = "Interpolated Peak 3 Index"
        output_rows.append(header)
        data_rows = rows[1:]

    rows_processed = 0
    rows_with_tof = 0

    for row_idx, row in enumerate(data_rows, start=2 if has_header else 1):
        row_out = list(row[:INTERP_PEAK3_COL_1_BASED])
        if len(row_out) < INTERP_PEAK3_COL_1_BASED:
            row_out.extend([""] * (INTERP_PEAK3_COL_1_BASED - len(row_out)))

        tof_value = ""
        peak1_idx = ""
        peak2_idx = ""
        interp_tof_value = ""
        interp_peak1_idx = ""
        interp_peak2_idx = ""
        interp_peak3_idx = ""
        try:
            signal_values = parse_signal_columns(row, SIGNAL_START_COL_1_BASED)
            (
                tof_value,
                peak1_idx,
                peak2_idx,
                interp_tof_value,
                interp_peak1_idx,
                interp_peak2_idx,
                interp_peak3_idx,
            ) = compute_tof_and_peaks(signal_values)
        except ValueError as exc:
            raise ValueError(f"Failed parsing signal at row {row_idx}: {exc}") from exc

        row_out[TOF_COL_1_BASED - 1] = tof_value
        row_out[PEAK1_COL_1_BASED - 1] = peak1_idx
        row_out[PEAK2_COL_1_BASED - 1] = peak2_idx
        row_out[INTERP_TOF_COL_1_BASED - 1] = interp_tof_value
        row_out[INTERP_PEAK1_COL_1_BASED - 1] = interp_peak1_idx
        row_out[INTERP_PEAK2_COL_1_BASED - 1] = interp_peak2_idx
        row_out[INTERP_PEAK3_COL_1_BASED - 1] = interp_peak3_idx
        if tof_value != "":
            rows_with_tof += 1

        output_rows.append(row_out)
        rows_processed += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as dest_file:
        writer = csv.writer(dest_file)
        writer.writerows(output_rows)

    return rows_processed, rows_with_tof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute TOF from signal columns 11+ and write raw/interpolated TOF and peak indexes to columns 10-16."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Input CSV file path (absolute or relative; relative also checks data/raw).",
    )
    parser.add_argument(
        "--header",
        choices=["auto", "yes", "no"],
        default="auto",
        help="Whether input CSV has a header row (default: auto).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional output CSV path. Ignored with --in-place.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input CSV directly.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.in_place and args.output_file is not None:
        raise ValueError("Use either --in-place or --output-file, not both.")

    input_csv = resolve_input_csv_path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_csv = build_output_path(
        input_csv=input_csv,
        output_file=args.output_file,
        in_place=args.in_place,
    )

    rows_processed, rows_with_tof = process_file(
        input_csv=input_csv,
        output_csv=output_csv,
        header_mode=args.header,
    )

    print(f"Rows processed: {rows_processed}")
    print(f"Rows with valid TOF: {rows_with_tof}")
    print(f"Output written to: {output_csv}")


if __name__ == "__main__":
    main()
