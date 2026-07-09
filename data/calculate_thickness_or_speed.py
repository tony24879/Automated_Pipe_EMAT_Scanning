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
    if mode == "thickness":
        return (col10_value / 2) * estimate

    if col10_value == 0:
        return ""
    return (estimate / 2) / col10_value


def build_output_path(input_csv: Path, mode: str) -> Path:
    suffix = "_thickness" if mode == "thickness" else "_speed"
    return PROCESSED_DIR / f"{input_csv.stem}{suffix}.csv"


def main() -> None:
    mode = prompt_mode()

    if mode == "thickness":
        estimate = prompt_float("Enter speed estimate: ")
    else:
        estimate = prompt_float("Enter tickness estimate: ")

    input_csv = prompt_input_file()
    output_csv = build_output_path(input_csv, mode)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    result_header = "Thickness" if mode == "thickness" else "Speed of Sound"
    rows_written = 0

    with input_csv.open("r", newline="", encoding="utf-8") as source_file, output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as dest_file:
        reader = csv.reader(source_file)
        writer = csv.writer(dest_file)

        first_row = next(reader, None)
        if first_row is None:
            writer.writerow(["Column 8", "Column 9", "Column 10", result_header])
            print(f"Input CSV is empty. Wrote header only to: {output_csv}")
            return

        has_header = False
        if len(first_row) >= 10:
            try:
                float(first_row[9])
            except ValueError:
                has_header = True

        if has_header:
            writer.writerow([first_row[7], first_row[8], first_row[9], result_header])
        else:
            writer.writerow(["Column 8", "Column 9", "Column 10", result_header])
            if len(first_row) >= 10:
                col10_value = float(first_row[9])
                result = compute_result(mode, estimate, col10_value)
                writer.writerow([first_row[7], first_row[8], first_row[9], result])
                rows_written += 1

        for row in reader:
            if len(row) < 10:
                continue

            try:
                col10_value = float(row[9])
            except ValueError:
                continue

            result = compute_result(mode, estimate, col10_value)
            writer.writerow([row[7], row[8], row[9], result])
            rows_written += 1

    print(f"Processed {rows_written} data rows.")
    print(f"Output written to: {output_csv}")


if __name__ == "__main__":
    main()
