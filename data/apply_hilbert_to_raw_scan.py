"""Apply Hilbert-envelope processing to raw scan CSV signal columns.

Input format expected from sync logger:
- Columns 1-9: pose/scan metadata
- Column 10: time of flight (will be recalculated)
- Columns 11+: signal samples (filtAscan)

For each row:
- Read signal samples from column 11 onward.
- Apply Hilbert envelope: abs(signal.hilbert(signal_values)).
- Recompute time of flight using the same logic as emat/sync_logger.py.
- Write output CSV beside the source file with "_hilbert" appended to filename.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np
from scipy import signal


def compute_time_of_flight(signal_values: List[float]) -> str:
    """Compute TOF from peak index difference in fixed sample windows."""
    if len(signal_values) <= 690:
        return ""

    first_start, first_end = 300, 380
    second_start, second_end = 610, 690

    first_window = signal_values[first_start:first_end + 1]
    second_window = signal_values[second_start:second_end + 1]

    if not first_window or not second_window:
        return ""

    first_peak_index = first_start + max(range(len(first_window)), key=first_window.__getitem__)
    second_peak_index = second_start + max(range(len(second_window)), key=second_window.__getitem__)

    sample_difference = second_peak_index - first_peak_index
    return str(sample_difference * (20e-9))


def parse_signal_columns(row: List[str]) -> List[float]:
    """Parse signal values from column 11 onward (1-based indexing)."""
    values: List[float] = []
    for cell in row[10:]:
        text = cell.strip()
        if text == "":
            continue
        values.append(float(text))
    return values


def process_file(input_path: Path, output_folder: Path | None = None) -> Path:
    """Process one input CSV and write a sibling *_hilbert.csv file."""
    if output_folder is None:
        output_path = input_path.with_name(f"{input_path.stem}_hilbert{input_path.suffix}")
    else:
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = output_folder / f"{input_path.stem}_hilbert{input_path.suffix}"

    with input_path.open("r", newline="", encoding="utf-8") as infile, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader, None)
        if header is None:
            raise ValueError(f"Input file is empty: {input_path}")

        signal_headers = header[10:]
        out_header = [*header[:9], "Time of Flight (s)", *signal_headers]
        writer.writerow(out_header)

        for row_idx, row in enumerate(reader, start=2):
            if not row:
                continue

            prefix = row[:9]
            try:
                raw_signal = parse_signal_columns(row)
            except ValueError as exc:
                raise ValueError(f"Failed parsing signal values at row {row_idx}: {exc}") from exc

            if not raw_signal:
                writer.writerow([*prefix, "", *row[10:]])
                continue

            raw_signal_np = np.asarray(raw_signal, dtype=float)
            hilbert_signal = np.abs(signal.hilbert(raw_signal_np))
            hilbert_signal_list = hilbert_signal.tolist()

            tof = compute_time_of_flight(hilbert_signal_list)
            writer.writerow([*prefix, tof, *hilbert_signal_list])

    return output_path


def resolve_input_path(raw_file: str) -> Path:
    """Resolve input path relative to data/raw when not absolute."""
    candidate = Path(raw_file)
    if candidate.is_absolute():
        return candidate

    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    return raw_dir / raw_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply Hilbert envelope to scan CSV signal columns and recalculate TOF."
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
        help="Optional output folder for the *_hilbert CSV.",
    )
    args = parser.parse_args()

    input_path = resolve_input_path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_path = process_file(input_path, output_folder=args.output_folder)
    print(f"Wrote Hilbert-processed CSV: {output_path}")


if __name__ == "__main__":
    main()
