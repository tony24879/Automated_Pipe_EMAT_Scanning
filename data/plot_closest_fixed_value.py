"""Find rows whose Theta/Axis column matches a fixed value most closely, then plot the rest.

Behavior:
- Reads columns 8, 9, 10 (1-based) from an input/working CSV.
- If --fixed-value-type is "theta", searches column 8 for the value closest to
  --fixed-value. If "axis", searches column 9 instead.
- Selects every row whose search-column value equals that closest value (there
  should be repeats), then plots the remaining two columns against each other
  as a line graph: for "theta" that's column 9 (x) vs column 10 (y); for
  "axis" that's column 8 (x) vs column 10 (y).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COL_THETA = 8  # 1-based
COL_AXIS = 9  # 1-based
COL_VALUE = 10  # 1-based


def resolve_input_csv_path(input_path: Path) -> Path:
    """Resolve input path with forgiving CSV conventions used in this repo."""
    candidates: list[Path] = [input_path]

    if input_path.suffix == "":
        candidates.append(input_path.with_suffix(".csv"))

    if not input_path.is_absolute():
        raw_dir = Path(__file__).resolve().parent / "raw"
        candidates.append(raw_dir / input_path.name)
        if input_path.suffix == "":
            candidates.append(raw_dir / f"{input_path.name}.csv")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find the Theta/Axis value closest to a fixed value, then plot the "
            "other two columns for all rows sharing that closest value."
        )
    )
    parser.add_argument("csv_file", type=Path, help="Path to input CSV file.")
    parser.add_argument(
        "--fixed-value-type",
        choices=["theta", "axis"],
        required=True,
        help="Which column to search: 'theta' (column 8) or 'axis' (column 9).",
    )
    parser.add_argument(
        "--fixed-value",
        type=float,
        required=True,
        help="Target numeric value to match as closely as possible.",
    )
    parser.add_argument(
        "--header",
        choices=["auto", "yes", "no"],
        default="auto",
        help="Whether CSV has a header row (default: auto).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional output image path. If omitted, figure is only shown.",
    )
    parser.add_argument(
        "--y-label",
        type=str,
        default="Column 10 value",
        help="Label for the y-axis (default: Column 10 value).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a figure window; useful for headless runs.",
    )
    parser.add_argument(
        "--override-tof-file",
        type=Path,
        default=None,
        help="Optional CSV/text file containing override column-10 values for all data rows.",
    )
    return parser.parse_args()


def _is_numeric(text: str) -> bool:
    try:
        float(text.strip())
        return True
    except ValueError:
        return False


def _has_header(rows: list[list[str]], check_col_idx: int, header_mode: str) -> bool:
    if header_mode == "yes":
        return True
    if header_mode == "no":
        return False

    if not rows:
        return False

    first_row = rows[0]
    if len(first_row) <= check_col_idx:
        return False

    return not _is_numeric(first_row[check_col_idx])


def load_override_values(path: Path) -> np.ndarray:
    values: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            for cell in row:
                text = cell.strip()
                if not text:
                    continue
                values.append(float(text))
    if not values:
        raise ValueError(f"No override values found in: {path}")
    return np.asarray(values, dtype=float)


def load_columns(
    csv_path: Path, header_mode: str, override_values: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load columns 8, 9, 10 (1-based) as float arrays of equal length.

    If override_values is given, it replaces column 10 (must match row count).
    """
    theta_idx, axis_idx, value_idx = COL_THETA - 1, COL_AXIS - 1, COL_VALUE - 1

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError(f"Input CSV is empty: {csv_path}")

    has_header = _has_header(rows, theta_idx, header_mode)
    data_rows = rows[1:] if has_header else rows

    theta_values: list[float] = []
    axis_values: list[float] = []
    tof_values: list[float] = []

    for row in data_rows:
        if len(row) <= value_idx:
            continue

        try:
            theta_value = float(row[theta_idx])
            axis_value = float(row[axis_idx])
            tof_value = float(row[value_idx])
        except ValueError:
            continue

        if override_values is None:
            tof_values.append(tof_value)

        theta_values.append(theta_value)
        axis_values.append(axis_value)

    if not theta_values:
        raise ValueError(f"No valid numeric rows found in {csv_path}.")

    if override_values is not None:
        if len(override_values) != len(theta_values):
            raise ValueError(
                "Override values length "
                f"({len(override_values)}) does not match valid data row count ({len(theta_values)})."
            )
        tof_array = np.asarray(override_values, dtype=float)
    else:
        tof_array = np.asarray(tof_values, dtype=float)

    return (
        np.asarray(theta_values, dtype=float),
        np.asarray(axis_values, dtype=float),
        tof_array,
    )


def find_closest_value_mask(search_column: np.ndarray, fixed_value: float) -> np.ndarray:
    """Return a boolean mask selecting all rows matching the closest value found."""
    closest_idx = int(np.argmin(np.abs(search_column - fixed_value)))
    closest_value = search_column[closest_idx]
    return search_column == closest_value


def main() -> None:
    args = parse_args()

    csv_path = resolve_input_csv_path(args.csv_file)
    if not csv_path.exists():
        raise SystemExit(f"Input CSV not found: {csv_path}")

    override_values = (
        load_override_values(args.override_tof_file) if args.override_tof_file is not None else None
    )
    theta_values, axis_values, tof_values = load_columns(csv_path, args.header, override_values)

    if args.fixed_value_type == "theta":
        search_column = theta_values
        x_values_all, x_label = axis_values, "Axis Position"
    else:
        search_column = axis_values
        x_values_all, x_label = theta_values, "Theta"

    mask = find_closest_value_mask(search_column, args.fixed_value)
    closest_value = search_column[mask][0]
    print(
        f"Closest {args.fixed_value_type} value to {args.fixed_value}: "
        f"{closest_value} ({int(mask.sum())} matching rows)"
    )

    x_values = x_values_all[mask]
    y_values = tof_values[mask]

    sort_order = np.argsort(x_values)
    x_values = x_values[sort_order]
    y_values = y_values[sort_order]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_values, y_values, marker="o")
    ax.set_xlabel(x_label)
    ax.set_ylabel(args.y_label)
    ax.set_title(
        f"{args.y_label} vs {x_label} at {args.fixed_value_type} = {closest_value}"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if args.save is not None:
        fig.savefig(args.save)
        print(f"Saved figure to {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
