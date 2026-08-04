"""Calculate averaged pairwise TOF values from waveform peaks in scan CSV files.

Input format expected from sync logger:
- Columns 1-9: pose/scan metadata
- Column 10: existing TOF (ignored and replaced)
- Columns 11+: waveform signal samples

For each row:
- Parse signal samples from column 11 onward.
- Detect peaks using the same method/constants as emat/sync_logger.py.
- Keep the first N detected peaks (`--num-peaks`).
- Compute TOF for each adjacent pair: (peak1, peak2), (peak2, peak3), ...
- Average the pairwise TOF values.
- Write output CSV with columns 1-9 from input and averaged TOF in column 10.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scipy.signal import find_peaks

RAW_DIR = Path(__file__).resolve().parent / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent / "processed"

# Keep these aligned with emat/sync_logger.py peak detection settings.
SKIP_SAMPLES = 150
MIN_PEAK_DISTANCE = 100
MIN_PROMINENCE = 500
TOF_SCALE_SECONDS = 20e-9


def resolve_input_path(raw_file: str) -> Path:
    candidate = Path(raw_file)
    if candidate.is_absolute():
        return candidate
    return RAW_DIR / candidate


def build_output_path(input_csv: Path, num_peaks: int, output_folder: Path | None = None) -> Path:
    folder = PROCESSED_DIR if output_folder is None else output_folder
    return folder / f"{input_csv.stem}_avg_tof_{num_peaks}{input_csv.suffix}"


def parse_signal_columns(row: list[str]) -> list[float]:
    """Parse signal values from column 11 onward (1-based indexing)."""
    values: list[float] = []
    for cell in row[10:]:
        text = cell.strip()
        if text == "":
            continue
        values.append(float(text))
    return values


def compute_avg_pairwise_tof(signal_values: list[float], num_peaks: int) -> str:
    """Return averaged adjacent-pair TOF in seconds, or empty string if unavailable."""
    if num_peaks < 2:
        raise ValueError("num_peaks must be at least 2.")

    if len(signal_values) <= SKIP_SAMPLES + 1:
        return ""

    post_noise = signal_values[SKIP_SAMPLES:]
    if not post_noise:
        return ""

    peak_indices, _ = find_peaks(
        post_noise,
        distance=MIN_PEAK_DISTANCE,
        prominence=MIN_PROMINENCE,
    )

    if len(peak_indices) < num_peaks:
        return ""

    selected_peaks = [SKIP_SAMPLES + int(idx) for idx in peak_indices[:num_peaks]]

    pairwise_tofs: list[float] = []
    for i in range(len(selected_peaks) - 1):
        sample_difference = selected_peaks[i + 1] - selected_peaks[i]
        pairwise_tofs.append(sample_difference * TOF_SCALE_SECONDS)

    if not pairwise_tofs:
        return ""

    avg_tof = sum(pairwise_tofs) / len(pairwise_tofs)
    return str(avg_tof)


def process_file(
    input_csv: Path,
    num_peaks: int,
    output_folder: Path | None = None,
    output_file: Path | None = None,
) -> Path:
    output_csv = (
        output_file
        if output_file is not None
        else build_output_path(input_csv, num_peaks=num_peaks, output_folder=output_folder)
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with input_csv.open("r", newline="", encoding="utf-8") as source_file, output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as dest_file:
        reader = csv.reader(source_file)
        writer = csv.writer(dest_file)

        header = next(reader, None)
        if header is None:
            raise ValueError(f"Input file is empty: {input_csv}")

        if len(header) < 11:
            raise ValueError(
                f"Input file must contain at least 11 columns (1-9 metadata, 10 TOF, 11+ signal): {input_csv}"
            )

        writer.writerow([*header[:9], "Average Time of Flight (s)"])

        rows_written = 0
        for row_idx, row in enumerate(reader, start=2):
            if len(row) < 11:
                continue

            try:
                signal_values = parse_signal_columns(row)
            except ValueError as exc:
                raise ValueError(f"Failed parsing signal values at row {row_idx}: {exc}") from exc

            avg_tof = compute_avg_pairwise_tof(signal_values, num_peaks=num_peaks)
            writer.writerow([*row[:9], avg_tof])
            rows_written += 1

    print(f"Processed {rows_written} data rows.")
    print(f"Output written to: {output_csv}")
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute averaged adjacent-pair TOF from first N detected peaks in CSV signal columns."
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
        "--output-folder",
        type=Path,
        default=None,
        help="Optional output folder for the *_avg_pairwise_tof CSV.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional explicit output CSV path. Overrides --output-folder.",
    )
    args = parser.parse_args()

    if args.num_peaks < 2:
        raise ValueError("--num-peaks must be at least 2.")

    input_csv = resolve_input_path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    process_file(
        input_csv,
        num_peaks=args.num_peaks,
        output_folder=args.output_folder,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
