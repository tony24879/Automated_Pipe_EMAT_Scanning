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
        type=Path,
        default=None,
        help="Optional custom output image path. Defaults to data/processed/plots/<csv_stem>_tof_histogram.png",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the figure window; only save the image.",
    )
    return parser.parse_args()


def _to_zero_based(index_1_based: int) -> int:
    if index_1_based < 1:
        raise ValueError("Column indices must be 1-based and >= 1.")
    return index_1_based - 1


def load_tof_values(csv_path: Path, tof_col: int) -> np.ndarray:
    tof_idx = _to_zero_based(tof_col)
    tof_vals: list[float] = []

    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) <= tof_idx:
                continue
            try:
                tof = float(row[tof_idx])
            except ValueError:
                # Skip headers and malformed rows.
                continue
            tof_vals.append(tof)

    if not tof_vals:
        raise ValueError(
            "No numeric TOF values found. Check the file path and the selected TOF column."
        )

    return np.array(tof_vals)


def default_export_path(csv_path: Path) -> Path:
    plots_dir = Path(__file__).resolve().parent / "processed" / "plots"
    return plots_dir / f"{csv_path.stem}_tof_histogram.png"


def plot_histogram(
    tof_vals: np.ndarray,
    bin_edges: np.ndarray,
    y_max: float,
    title: str,
    save_path: Path,
) -> None:

    plt.figure(figsize=(9, 5.5))
    plt.hist(tof_vals, bins=bin_edges, color="steelblue", edgecolor="black", alpha=0.85)
    plt.xlim(float(bin_edges[0]), float(bin_edges[-1]))
    plt.ylim(0.0, y_max)
    plt.xlabel("Time of Flight (s)")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

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

    datasets: list[tuple[Path, np.ndarray]] = []
    for csv_path in csv_files:
        tof_vals = load_tof_values(csv_path, args.tof_col)
        datasets.append((csv_path, tof_vals))

    global_min = float(min(np.min(tof_vals) for _, tof_vals in datasets))
    global_max = float(max(np.max(tof_vals) for _, tof_vals in datasets))
    if np.isclose(global_min, global_max):
        global_max = global_min + 1e-12

    bin_edges = np.linspace(global_min, global_max, args.bins + 1)

    global_y_max = 0.0
    for _, tof_vals in datasets:
        counts, _ = np.histogram(tof_vals, bins=bin_edges)
        global_y_max = max(global_y_max, float(np.max(counts)))
    y_max = max(1.0, global_y_max * 1.05)

    if args.save is not None and len(datasets) > 1 and args.save.suffix:
        raise ValueError(
            "When plotting multiple CSV files, --save must be a directory path (no file extension)."
        )

    for csv_path, tof_vals in datasets:
        if args.save is None:
            save_path = default_export_path(csv_path)
        elif len(datasets) == 1 and args.save.suffix:
            save_path = args.save
        else:
            save_path = args.save / f"{csv_path.stem}_tof_histogram.png"

        title = f"Time of Flight Histogram\n{csv_path.name}"
        plot_histogram(
            tof_vals=tof_vals,
            bin_edges=bin_edges,
            y_max=y_max,
            title=title,
            save_path=save_path,
        )

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
