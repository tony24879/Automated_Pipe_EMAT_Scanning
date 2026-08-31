"""Plot Time of Flight values from CSV files as a scatter plot with a line of best fit.

Default behavior:
- Reads TOF values from column 10 (1-based)
- x-axis is the index of the value within its dataset, y-axis is the value itself
- Accepts one or more CSV files and plots them on the same axes, each with its own
  color and line of best fit
- Saves output image to data/processed/plots when requested
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from repeat_point_grouping import (
        grouped_tof_values,
        group_rows_by_repeat_point,
        group_row_indices_by_repeat_point,
    )
except ModuleNotFoundError:
    from data.repeat_point_grouping import (
        grouped_tof_values,
        group_rows_by_repeat_point,
        group_row_indices_by_repeat_point,
    )


def _path_or_none(value: str | None) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value)


def _to_zero_based(index_1_based: int) -> int:
    if index_1_based < 1:
        raise ValueError("Column indices must be 1-based and >= 1.")
    return index_1_based - 1


def load_tof_values(csv_path: Path, tof_col: int) -> np.ndarray:
    tof_idx = _to_zero_based(tof_col)
    values: list[float] = []

    with csv_path.open("r", newline="", encoding="utf-8") as source_file:
        reader = csv.reader(source_file)
        for row in reader:
            if not row:
                continue
            if len(row) <= tof_idx:
                continue
            raw_value = row[tof_idx].strip()
            if not raw_value:
                continue
            try:
                values.append(float(raw_value))
            except ValueError:
                continue

    if not values:
        raise ValueError(f"No numeric TOF values found in {csv_path} with column {tof_col}.")

    return np.asarray(values, dtype=float)


def _read_rows(csv_path: Path) -> list[list[str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as source_file:
        return [row for row in csv.reader(source_file) if row]


def load_grouped_tof_values(
    csv_path: Path,
    tof_col: int,
    override_values: np.ndarray | None = None,
) -> list[np.ndarray]:
    rows = _read_rows(csv_path)
    grouped_rows = group_rows_by_repeat_point(rows, tof_col=tof_col, key_cols=(8, 9), skip_header=True)
    if not grouped_rows:
        raise ValueError(
            f"No grouped numeric TOF values found in {csv_path} using columns 8/9 change detection."
        )

    if override_values is None:
        return [np.asarray(group, dtype=float) for group in grouped_tof_values(grouped_rows, tof_col=tof_col)]

    group_indices = group_row_indices_by_repeat_point(rows, tof_col=tof_col, key_cols=(8, 9), skip_header=True)
    required_count = sum(len(indices) for indices in group_indices)
    if len(override_values) != required_count:
        raise ValueError(
            "Override TOF length does not match grouped data row count. "
            f"Expected {required_count}, got {len(override_values)}."
        )

    override_array = np.asarray(override_values, dtype=float)
    return [override_array[indices] for indices in group_indices]


def load_override_tof_values(path: Path) -> np.ndarray:
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
        raise ValueError(f"No override TOF values found in: {path}")
    return np.asarray(values, dtype=float)


def default_export_path(csv_path: Path) -> Path:
    plots_dir = Path(__file__).resolve().parent / "processed" / "plots"
    return plots_dir / f"{csv_path.stem}_scatter.png"


def plot_scatter(
    datasets: list[tuple[str, np.ndarray]],
    title: str,
    y_label: str,
    save_path: Path | None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (label, values) in enumerate(datasets):
        color = color_cycle[i % len(color_cycle)]
        indices = np.arange(len(values))
        ax.scatter(indices, values, color=color, alpha=0.7, s=18, label=label)

        if len(values) >= 2:
            slope, intercept = np.polyfit(indices, values, 1)
            fit_line = slope * indices + intercept
            ax.plot(indices, fit_line, color=color, linewidth=2, linestyle="--")

    ax.set_xlabel("Index")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    if len(datasets) > 1:
        ax.legend()
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved scatter plot image to: {save_path}")


def _slugify(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Time of Flight values from one or more CSV files as an index-vs-value scatter plot."
    )
    parser.add_argument(
        "csv_files",
        type=Path,
        nargs="+",
        help="One or more input CSV files to compare on the same scatter axis.",
    )
    parser.add_argument(
        "--tof-col",
        type=int,
        default=10,
        help="1-based column index for TOF values (default: 10).",
    )
    parser.add_argument(
        "--save",
        type=_path_or_none,
        default="",
        help="Optional custom output image path. If blank or omitted, no image is saved.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Time of Flight Scatter Plot",
        help="Title to display above the scatter plot.",
    )
    parser.add_argument(
        "--y-label",
        type=str,
        default="Time of Flight (s)",
        help="Label for the y-axis.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the figure window; only save the image.",
    )
    parser.add_argument(
        "--override-tof-file",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Optional CSV/text file(s) containing override TOF values for the selected data rows, "
            "one per input CSV (in the same order)."
        ),
    )
    parser.add_argument(
        "--group-by-repeat-point",
        action="store_true",
        help=(
            "Group rows when column 8 or 9 changes and plot each group as its own "
            "dataset with its own color and line of best fit."
        ),
    )
    parser.add_argument(
        "--separate-plots",
        action="store_true",
        help=(
            "Plot each dataset (each CSV, or each group when --group-by-repeat-point is set) "
            "on its own figure instead of combining them onto a single plot."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for csv_path in args.csv_files:
        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    if args.override_tof_file is not None and len(args.override_tof_file) != len(args.csv_files):
        raise ValueError(
            "Number of --override-tof-file entries must match the number of input CSV files."
        )
    override_values_by_csv: list[np.ndarray | None] = (
        [load_override_tof_values(path) for path in args.override_tof_file]
        if args.override_tof_file is not None
        else [None] * len(args.csv_files)
    )

    datasets: list[tuple[str, np.ndarray]] = []

    for csv_path, override_values in zip(args.csv_files, override_values_by_csv):
        if args.group_by_repeat_point:
            groups = load_grouped_tof_values(csv_path, args.tof_col, override_values=override_values)
            for idx, group_values in enumerate(groups, start=1):
                label = f"{csv_path.stem} - Group {idx}" if len(args.csv_files) > 1 else f"Group {idx}"
                datasets.append((label, group_values))
        else:
            if override_values is not None:
                values = override_values
            else:
                values = load_tof_values(csv_path, args.tof_col)
            datasets.append((csv_path.stem, values))

    title = args.title.strip() if args.title and args.title.strip() else "Time of Flight Scatter Plot"
    y_label = args.y_label.strip() if args.y_label and args.y_label.strip() else "Time of Flight (s)"

    save_arg: Path | None = None
    if args.save not in (None, ""):
        save_arg = Path(args.save)

    if args.separate_plots:
        for label, values in datasets:
            dataset_title = f"{title}\n{label}" if len(datasets) > 1 else title

            if save_arg is None:
                dataset_save_path = None
            elif save_arg.suffix:
                dataset_save_path = save_arg.parent / f"{save_arg.stem}_{_slugify(label)}{save_arg.suffix}"
            else:
                dataset_save_path = save_arg / f"{_slugify(label)}_scatter.png"

            plot_scatter([(label, values)], title=dataset_title, y_label=y_label, save_path=dataset_save_path)
    else:
        save_path: Path | None = None
        if save_arg is not None:
            save_path = save_arg
            if save_path.suffix == "":
                save_path = save_path / f"{args.csv_files[0].stem}_scatter.png"
        elif len(args.csv_files) == 1 and not args.group_by_repeat_point:
            save_path = default_export_path(args.csv_files[0])

        plot_scatter(datasets, title=title, y_label=y_label, save_path=save_path)

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
