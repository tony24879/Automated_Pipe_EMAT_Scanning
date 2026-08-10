"""Recalculate row-wise TOF from waveform columns and write into column 10.

Behavior:
- Reads an input CSV from an absolute path or from data/raw when relative.
- For each data row, parses waveform samples from column 11 onward (1-based).
- Detects peaks using the same constants as data/plot_row_peaks.py.
- Computes TOF as in emat/sync_logger.py: (second_peak - first_peak) * 20e-9.
- Writes output CSV columns 1-12 where:
    - Column 10 is recalculated TOF.
    - Column 11 is first peak index.
    - Column 12 is second peak index.
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


def compute_tof_and_peaks(signal_values: list[float]) -> tuple[str, str, str]:
    """Return TOF and first two peak indices, or empty strings if unavailable."""
    if len(signal_values) <= SKIP_SAMPLES + 1:
        return "", "", ""

    post_noise = signal_values[SKIP_SAMPLES:]
    acf = np.correlate(post_noise, post_noise, mode='full')
    acf = acf[acf.size // 2:]
    if not post_noise:
        return "", "", ""

    peak_indices, _ = find_peaks(
        acf,
        distance=MIN_PEAK_DISTANCE,
        prominence=MIN_PROMINENCE,
    )

    if len(peak_indices) < 2:
        return "", "", ""

    #first_peak = SKIP_SAMPLES + int(peak_indices[0])
    first_peak = int(peak_indices[0])
    #second_peak = SKIP_SAMPLES + int(peak_indices[1])
    second_peak = int(peak_indices[1])
    #sample_difference = second_peak - first_peak
    sample_difference = second_peak
    return str(sample_difference * TOF_SCALE_SECONDS), str(first_peak), str(second_peak)


def build_output_path(input_csv: Path, output_file: Path | None, in_place: bool) -> Path:
    if in_place:
        return input_csv
    if output_file is not None:
        return output_file
    return PROCESSED_DIR / f"{input_csv.stem}_recalc_tof_peaks_col10_12{input_csv.suffix}"


def process_file(input_csv: Path, output_csv: Path, header_mode: str) -> tuple[int, int]:
    """Recalculate TOF for each data row and write TOF/peak indices to columns 10-12."""
    with input_csv.open("r", newline="", encoding="utf-8") as source_file:
        rows = list(csv.reader(source_file))

    if not rows:
        raise ValueError(f"Input CSV is empty: {input_csv}")

    start_col_idx = SIGNAL_START_COL_1_BASED - 1
    has_header = _has_header(rows, start_col_idx, header_mode)

    output_rows: list[list[str]] = []
    data_rows = rows
    if has_header:
        header = list(rows[0][:PEAK2_COL_1_BASED])
        if len(header) < PEAK2_COL_1_BASED:
            header.extend([""] * (PEAK2_COL_1_BASED - len(header)))
        header[TOF_COL_1_BASED - 1] = "Time of Flight (s)"
        header[PEAK1_COL_1_BASED - 1] = "Peak 1 Index"
        header[PEAK2_COL_1_BASED - 1] = "Peak 2 Index"
        output_rows.append(header)
        data_rows = rows[1:]

    rows_processed = 0
    rows_with_tof = 0

    for row_idx, row in enumerate(data_rows, start=2 if has_header else 1):
        row_out = list(row[:PEAK2_COL_1_BASED])
        if len(row_out) < PEAK2_COL_1_BASED:
            row_out.extend([""] * (PEAK2_COL_1_BASED - len(row_out)))

        tof_value = ""
        peak1_idx = ""
        peak2_idx = ""
        try:
            signal_values = parse_signal_columns(row, SIGNAL_START_COL_1_BASED)
            tof_value, peak1_idx, peak2_idx = compute_tof_and_peaks(signal_values)
        except ValueError as exc:
            raise ValueError(f"Failed parsing signal at row {row_idx}: {exc}") from exc

        row_out[TOF_COL_1_BASED - 1] = tof_value
        row_out[PEAK1_COL_1_BASED - 1] = peak1_idx
        row_out[PEAK2_COL_1_BASED - 1] = peak2_idx
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
            "Recompute TOF from signal columns 11+ and write TOF/peak indices to columns 10-12."
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
