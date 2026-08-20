"""Plot a vertical box-and-whisker chart of Time of Flight values from CSV files.

Default behavior:
- Reads TOF values from column 10 (1-based)
- Accepts one or more CSV files and plots them on the same y-axis
- Saves output image to data/processed/plots when requested
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from repeat_point_grouping import grouped_tof_values, group_rows_by_repeat_point
except ModuleNotFoundError:
    from data.repeat_point_grouping import grouped_tof_values, group_rows_by_repeat_point


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

    required_count = sum(len(group) for group in grouped_rows)
    if len(override_values) != required_count:
        raise ValueError(
            "Override TOF length does not match grouped data row count. "
            f"Expected {required_count}, got {len(override_values)}."
        )

    grouped_values: list[np.ndarray] = []
    cursor = 0
    for group in grouped_rows:
        count = len(group)
        grouped_values.append(np.asarray(override_values[cursor : cursor + count], dtype=float))
        cursor += count
    return grouped_values


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
    return plots_dir / f"{csv_path.stem}_boxplot.png"


def plot_boxplot(
    datasets: list[tuple[Path, np.ndarray]],
    title: str,
    y_label: str,
    save_path: Path | None,
) -> None:
    labels = [csv_path.stem for csv_path, _ in datasets]
    data = [values for _, values in datasets]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(
        data,
        vert=True,
        patch_artist=True,
        labels=labels,
        widths=0.5,
        medianprops={"color": "black"},
        boxprops={"color": "black"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
    )

    for box in bp["boxes"]:
        box.set(facecolor="lightsteelblue", alpha=0.85)

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved box plot image to: {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a vertical box-and-whisker chart of Time of Flight values from one or more CSV files."
    )
    parser.add_argument(
        "csv_files",
        type=Path,
        nargs="+",
        help="One or more input CSV files to compare on the same boxplot axis.",
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
        default="Time of Flight Box Plot",
        help="Title to display above the box plot.",
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
        default=None,
        help="Optional CSV/text file containing override TOF values for the selected data rows.",
    )
    parser.add_argument(
        "--group-by-repeat-point",
        action="store_true",
        help=(
            "Group rows when column 8 or 9 changes and generate one box plot per group "
            "for each input CSV."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for csv_path in args.csv_files:
        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    override_values = load_override_tof_values(args.override_tof_file) if args.override_tof_file is not None else None
    if override_values is not None and len(args.csv_files) > 1:
        raise ValueError("Override TOF values are only supported for a single plotted CSV.")

    datasets: list[tuple[Path, np.ndarray]] = []
    grouped_datasets: list[tuple[Path, list[np.ndarray]]] = []

    for csv_path in args.csv_files:
        if args.group_by_repeat_point:
            grouped_datasets.append(
                (csv_path, load_grouped_tof_values(csv_path, args.tof_col, override_values=override_values))
            )
            continue

        if override_values is not None:
            datasets.append((csv_path, override_values))
        else:
            datasets.append((csv_path, load_tof_values(csv_path, args.tof_col)))

    title = args.title.strip() if args.title and args.title.strip() else "Time of Flight Box Plot"
    y_label = args.y_label.strip() if args.y_label and args.y_label.strip() else "Time of Flight (s)"

    save_path: Path | None = None
    if args.save not in (None, ""):
        save_path = Path(args.save)
        if save_path.suffix == "" and not args.group_by_repeat_point:
            save_path = save_path / f"{datasets[0][0].stem}_boxplot.png"

    if not args.group_by_repeat_point and save_path is None and len(datasets) == 1:
        save_path = default_export_path(datasets[0][0])

    if args.group_by_repeat_point:
        for csv_path, groups in grouped_datasets:
            for idx, group_values in enumerate(groups, start=1):
                group_title = f"{title}\n{csv_path.name} - Group {idx}"
                group_dataset = [(csv_path, group_values)]

                group_save_path: Path | None = None
                if save_path is not None:
                    if save_path.suffix:
                        group_save_path = save_path.parent / f"{save_path.stem}_{csv_path.stem}_group_{idx}{save_path.suffix}"
                    else:
                        group_save_path = save_path / f"{csv_path.stem}_group_{idx}_boxplot.png"

                plot_boxplot(group_dataset, title=group_title, y_label=y_label, save_path=group_save_path)
    else:
        plot_boxplot(datasets, title=title, y_label=y_label, save_path=save_path)

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
