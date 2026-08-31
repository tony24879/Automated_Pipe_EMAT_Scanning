"""Plot a histogram of Time of Flight values from a scan CSV.

Default behavior:
- Reads TOF values from column 10 (1-based)
- Saves output image to data/processed/plots
"""

from __future__ import annotations

import argparse
import csv
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot histogram(s) of Time of Flight values from one or more CSV files."
    )
    parser.add_argument(
        "csv_files",
        type=Path,
        nargs="+",
        help="Path(s) to input CSV file(s) (for example: data/raw/sync_scan_20260701_121153.csv)",
    )
    parser.add_argument(
        "--tof-col",
        type=int,
        default=10,
        help="1-based column index for time of flight values (default: 10)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: 50)",
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
        default="Time of Flight Histogram",
        help="Title to show above the histogram (default: Time of Flight Histogram)",
    )
    parser.add_argument(
        "--x-label",
        type=str,
        default="Time of Flight (s)",
        help="Label for the x-axis (default: Time of Flight (s))",
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
            "Group rows when column 8 or 9 changes and generate one histogram per group "
            "for each input CSV."
        ),
    )
    return parser.parse_args()


def _to_zero_based(index_1_based: int) -> int:
    if index_1_based < 1:
        raise ValueError("Column indices must be 1-based and >= 1.")
    return index_1_based - 1


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


def load_tof_values(csv_path: Path, tof_col: int, override_values: np.ndarray | None = None) -> np.ndarray:
    if override_values is not None:
        return override_values

    tof_idx = _to_zero_based(tof_col)
    tof_vals: list[float] = []

    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) <= tof_idx:
                continue
            raw_value = row[tof_idx].strip()
            if not raw_value:
                continue
            try:
                tof = float(raw_value)
            except ValueError:
                # Skip headers and malformed rows.
                continue
            tof_vals.append(tof)

    if not tof_vals:
        raise ValueError(
            "No numeric TOF values found. Check the file path and the selected TOF column."
        )

    return np.array(tof_vals)


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
    grouped_values: list[np.ndarray] = [override_array[indices] for indices in group_indices]
    return grouped_values


def default_export_path(csv_path: Path) -> Path:
    plots_dir = Path(__file__).resolve().parent / "processed" / "plots"
    return plots_dir / f"{csv_path.stem}_histogram.png"


def plot_histogram(
    tof_vals: np.ndarray,
    bin_edges: np.ndarray,
    y_max: float,
    title: str,
    x_label: str,
    save_path: Path | None,
) -> None:

    plt.figure(figsize=(9, 5.5))
    plt.hist(tof_vals, bins=bin_edges, color="steelblue", edgecolor="black", alpha=0.85)
    plt.xlim(float(bin_edges[0]), float(bin_edges[-1]))
    plt.ylim(0.0, y_max)
    plt.xlabel(x_label)
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved histogram image to: {save_path}")


def main() -> None:
    args = parse_args()

    if args.bins < 1:
        raise ValueError("--bins must be >= 1.")

    csv_files = args.csv_files
    for csv_path in csv_files:
        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    if args.override_tof_file is not None and len(args.override_tof_file) != len(csv_files):
        raise ValueError(
            "Number of --override-tof-file entries must match the number of input CSV files."
        )
    override_values_by_csv: list[np.ndarray | None] = (
        [load_override_tof_values(path) for path in args.override_tof_file]
        if args.override_tof_file is not None
        else [None] * len(csv_files)
    )

    save_arg: Path | None = None
    if args.save not in (None, ""):
        save_arg = Path(args.save)

    datasets: list[tuple[Path, np.ndarray]] = []
    grouped_datasets: list[tuple[Path, list[np.ndarray]]] = []
    for csv_path, override_values in zip(csv_files, override_values_by_csv):
        if args.group_by_repeat_point:
            grouped_datasets.append(
                (csv_path, load_grouped_tof_values(csv_path, args.tof_col, override_values=override_values))
            )
        else:
            tof_vals = load_tof_values(csv_path, args.tof_col, override_values=override_values)
            datasets.append((csv_path, tof_vals))

    if args.group_by_repeat_point:
        for csv_path, groups in grouped_datasets:
            for index, tof_vals in enumerate(groups, start=1):
                local_min = float(np.min(tof_vals))
                local_max = float(np.max(tof_vals))
                if np.isclose(local_min, local_max):
                    local_max = local_min + 1e-12

                bin_edges = np.linspace(local_min, local_max, args.bins + 1)
                counts, _ = np.histogram(tof_vals, bins=bin_edges)
                y_max = max(1.0, float(np.max(counts)) * 1.05)

                if save_arg is None:
                    save_path = None
                elif save_arg.suffix:
                    save_path = save_arg.parent / f"{save_arg.stem}_{csv_path.stem}_group_{index}{save_arg.suffix}"
                else:
                    save_path = save_arg / f"{csv_path.stem}_group_{index}_histogram.png"

                title = args.title.strip() if args.title and args.title.strip() else "Time of Flight Histogram"
                title = f"{title}\n{csv_path.name} - Group {index}"
                plot_histogram(
                    tof_vals=tof_vals,
                    bin_edges=bin_edges,
                    y_max=y_max,
                    title=title,
                    x_label=args.x_label.strip() if args.x_label and args.x_label.strip() else "Time of Flight (s)",
                    save_path=save_path,
                )

        if args.no_show:
            plt.close("all")
        else:
            plt.show()
        return

    # Shared bin edges across all datasets make distribution comparisons
    # meaningful when plotting multiple scans in one run.
    global_min = float(min(np.min(tof_vals) for _, tof_vals in datasets))
    global_max = float(max(np.max(tof_vals) for _, tof_vals in datasets))
    if np.isclose(global_min, global_max):
        global_max = global_min + 1e-12

    bin_edges = np.linspace(global_min, global_max, args.bins + 1)

    # Keep y-axis consistent across outputs so bar heights are comparable.
    global_y_max = 0.0
    for _, tof_vals in datasets:
        counts, _ = np.histogram(tof_vals, bins=bin_edges)
        global_y_max = max(global_y_max, float(np.max(counts)))
    y_max = max(1.0, global_y_max * 1.05)

    for csv_path, tof_vals in datasets:
        if save_arg is None:
            save_path = None
        elif len(datasets) == 1 and save_arg.suffix:
            save_path = save_arg
        else:
            save_dir = save_arg if (not save_arg.suffix or save_arg.is_dir()) else save_arg.parent
            save_path = save_dir / f"{csv_path.stem}_histogram.png"

        title = args.title.strip() if args.title and args.title.strip() else "Time of Flight Histogram"
        if len(datasets) > 1:
            title = f"{title}\n{csv_path.name}"
        plot_histogram(
            tof_vals=tof_vals,
            bin_edges=bin_edges,
            y_max=y_max,
            title=title,
            x_label=args.x_label.strip() if args.x_label and args.x_label.strip() else "Time of Flight (s)",
            save_path=save_path,
        )

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
