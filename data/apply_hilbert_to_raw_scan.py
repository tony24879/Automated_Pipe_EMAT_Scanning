"""Apply optional Hilbert processing to raw scan CSV signal columns.

For each row:
- Read signal samples from column 11 onward (1-based indexing).
- If mode is "none", write the original signal samples back unchanged.
- If mode is "hilbert", write abs(hilbert(signal_values)).
- Preserve columns 1-10 unchanged and keep the output CSV shape consistent.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import signal


def parse_signal_columns(row: list[str]) -> list[float]:
    """Parse signal values from column 11 onward (1-based indexing)."""
    values: list[float] = []
    for cell in row[10:]:
        text = cell.strip()
        if text == "":
            continue
        values.append(float(text))
    return values


def process_file(input_path: Path, output_file: Path | None = None, filter_mode: str = "hilbert") -> Path:
    """Process one input CSV and write a filtered output CSV without changing metadata columns."""
    mode = filter_mode.lower()
    if mode not in {"none", "hilbert"}:
        raise ValueError(f"Unsupported filter mode: {filter_mode!r}")

    if output_file is None:
        output_path = input_path.with_name(f"{input_path.stem}_processed{input_path.suffix}")
    else:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_path = output_file

    if input_path.resolve() == output_path.resolve():
        temp_path = output_path.with_name(f"{output_path.stem}_tmp{output_path.suffix}")
    else:
        temp_path = output_path

    with input_path.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Input file is empty: {input_path}")

    processed_rows: list[list[str]] = []
    header = rows[0]
    processed_rows.append(header)

    for row_idx, row in enumerate(rows[1:], start=2):
        if not row:
            processed_rows.append(row)
            continue

        prefix = row[:10]
        try:
            raw_signal = parse_signal_columns(row)
        except ValueError as exc:
            raise ValueError(f"Failed parsing signal values at row {row_idx}: {exc}") from exc

        if not raw_signal:
            processed_rows.append(row)
            continue

        signal_values = np.asarray(raw_signal, dtype=float)
        if mode == "hilbert":
            filtered_signal = np.abs(signal.hilbert(signal_values))
        else:
            filtered_signal = signal_values

        processed_rows.append([*prefix, *filtered_signal.tolist()])

    with temp_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(processed_rows)

    if temp_path != output_path:
        temp_path.replace(output_path)

    return output_path


def resolve_input_path(raw_file: str) -> Path:
    """Resolve input paths in a forgiving way for repo-relative and raw-dir paths."""
    candidate = Path(raw_file)
    if candidate.is_absolute():
        return candidate

    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"

    candidates = [
        project_root / candidate,
        raw_dir / candidate,
        raw_dir / candidate.name,
        candidate,
    ]

    for possible in candidates:
        if possible.exists():
            return possible

    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a no-op or Hilbert filter to scan CSV signal columns."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="sync_scan_20260709_112954.csv",
        help="Input CSV file name in data/raw (or an absolute path).",
    )
    parser.add_argument(
        "--mode",
        choices=["none", "hilbert"],
        default="hilbert",
        help="Filter mode to apply to columns 11+ (default: hilbert).",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=None,
        help="Optional output folder for the processed CSV.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional full output CSV path. Overrides --output-folder when set.",
    )
    args = parser.parse_args()

    input_path = resolve_input_path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    if args.output_folder is not None and args.output_file is not None:
        raise ValueError("Use either --output-folder or --output-file, not both.")

    output_target = args.output_file
    if output_target is None and args.output_folder is not None:
        output_target = args.output_folder / f"{input_path.stem}_processed{input_path.suffix}"

    output_path = process_file(input_path, output_file=output_target, filter_mode=args.mode)
    print(f"Wrote filtered CSV: {output_path}")


if __name__ == "__main__":
    main()
