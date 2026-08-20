"""Compute thickness or speed-of-sound from column-10 TOF-style values.

Modes:
- thickness: result = (col10 / 2) * speed_estimate
- speed:     result = (thickness_estimate * 2) / col10
"""

import argparse
import csv
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent / "processed"


def prompt_mode() -> str:
    while True:
        mode = input("Calculate 'thickness' or 'speed'? ").strip().lower()
        if mode in {"thickness", "speed"}:
            return mode
        print("Please enter either 'thickness' or 'speed'.")


def prompt_float(prompt_text: str) -> float:
    while True:
        raw_value = input(prompt_text).strip()
        try:
            return float(raw_value)
        except ValueError:
            print("Please enter a valid numeric value.")


def prompt_input_file() -> Path:
    while True:
        file_input = input(
            "Enter CSV filename from raw folder (for example: sync_scan_20260709_112954.csv): "
        ).strip()
        if not file_input:
            print("Filename cannot be empty.")
            continue

        file_path = Path(file_input)
        if not file_path.is_absolute():
            file_path = RAW_DIR / file_path

        if file_path.exists() and file_path.is_file() and file_path.suffix.lower() == ".csv":
            return file_path

        print(f"CSV file not found: {file_path}")


def compute_result(mode: str, estimate: float, col10_value: float):
    """Apply the selected two-way-travel conversion formula."""
    if mode == "thickness":
        # Divide by 2 for one-way travel distance before scaling by speed.
        return (col10_value / 2) * estimate

    if col10_value == 0:
        return ""
    # Rearranged from thickness = (tof/2) * speed.
    return (estimate * 2) / col10_value


def calculate_thickness_or_speed_values(input_csv: Path, mode: str, estimate: float) -> list[float]:
    """Return calculated thickness/speed values for each numeric column-10 entry in the CSV."""
    results: list[float] = []

    with input_csv.open("r", newline="", encoding="utf-8") as source_file:
        reader = csv.reader(source_file)
        first_row = next(reader, None)
        if first_row is None:
            return results

        if len(first_row) >= 10:
            try:
                first_value = float(first_row[9])
            except ValueError:
                pass
            else:
                computed = compute_result(mode, estimate, first_value)
                if computed != "":
                    results.append(float(computed))

        for row in reader:
            if len(row) < 10:
                continue
            try:
                col10_value = float(row[9])
            except ValueError:
                continue
            computed = compute_result(mode, estimate, col10_value)
            if computed != "":
                results.append(float(computed))

    return results


def build_output_path(input_csv: Path, mode: str) -> Path:
    suffix = "_thickness" if mode == "thickness" else "_speed"
    return PROCESSED_DIR / f"{input_csv.stem}{suffix}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate thickness or speed from column-10 values in a CSV."
    )
    parser.add_argument(
        "--mode",
        choices=["thickness", "speed"],
        default=None,
        help="Calculation mode. If omitted, script prompts interactively.",
    )
    parser.add_argument(
        "--estimate",
        type=float,
        default=None,
        help="Other parameter estimate (speed estimate for thickness mode, thickness estimate for speed mode).",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Input CSV path. If relative, it is resolved against data/raw.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=None,
        help="Optional output folder. Defaults to data/processed.",
    )
    return parser.parse_args()


def resolve_input_csv(input_csv: Path | None) -> Path | None:
    if input_csv is None:
        return None
    if input_csv.is_absolute():
        return input_csv
    return RAW_DIR / input_csv


def main() -> None:
    args = parse_args()

    mode = args.mode if args.mode is not None else prompt_mode()

    if args.estimate is not None:
        estimate = float(args.estimate)
    else:
        if mode == "thickness":
            estimate = prompt_float("Enter speed estimate: ")
        else:
            estimate = prompt_float("Enter tickness estimate: ")

    input_csv = resolve_input_csv(args.input_csv)
    if input_csv is None:
        input_csv = prompt_input_file()
    elif not (input_csv.exists() and input_csv.is_file() and input_csv.suffix.lower() == ".csv"):
        raise FileNotFoundError(f"CSV file not found: {input_csv}")

    results = calculate_thickness_or_speed_values(input_csv, mode, estimate)
    print(results)
    print(f"Processed {len(results)} calculated values.")


if __name__ == "__main__":
    main()
