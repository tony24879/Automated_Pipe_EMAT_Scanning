"""Plot a Time of Flight heatmap from a scan CSV in data/raw.

Expected default columns (1-based):
- 8: Theta (deg)
- 9: Axis Position (mm)
- 10: Time of Flight (s)
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a Time of Flight heatmap from a CSV file.")
    parser.add_argument(
        "csv_file",
        type=Path,
        help="Path to the CSV file (for example: data/raw/sync_scan_20260701_121153.csv)",
    )
    parser.add_argument(
        "--theta-col",
        type=int,
        default=8,
        help="1-based column index for theta in degrees (default: 8)",
    )
    parser.add_argument(
        "--x-col",
        type=int,
        default=9,
        help="1-based column index for x axis position in mm (default: 9)",
    )
    parser.add_argument(
        "--tof-col",
        type=int,
        default=10,
        help="1-based column index for time of flight in seconds (default: 10)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional output image path. If omitted, only a window is shown.",
    )
    return parser.parse_args()


def _to_zero_based(index_1_based: int) -> int:
    if index_1_based < 1:
        raise ValueError("Column indices must be 1-based and >= 1.")
    return index_1_based - 1


def load_points(csv_path: Path, theta_col: int, x_col: int, tof_col: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_idx = _to_zero_based(theta_col)
    x_idx = _to_zero_based(x_col)
    tof_idx = _to_zero_based(tof_col)

    theta_vals: list[float] = []
    x_vals: list[float] = []
    tof_vals: list[float] = []

    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            max_idx = max(theta_idx, x_idx, tof_idx)
            if len(row) <= max_idx:
                continue
            try:
                theta = float(row[theta_idx])
                x = float(row[x_idx])
                tof = float(row[tof_idx])
            except ValueError:
                # Skip header or malformed rows.
                continue

            theta_vals.append(theta)
            x_vals.append(x)
            tof_vals.append(tof)

    if not theta_vals:
        raise ValueError(
            "No numeric data points found. Check file path and column indices for theta/x/tof."
        )

    return np.array(theta_vals), np.array(x_vals), np.array(tof_vals)


def build_heatmap_grid(theta: np.ndarray, x: np.ndarray, tof: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Average duplicate points at identical (theta, x) coordinates.
    buckets: dict[tuple[float, float], list[float]] = defaultdict(list)
    for t_val, x_val, tof_val in zip(theta, x, tof):
        buckets[(t_val, x_val)].append(tof_val)

    unique_theta = np.array(sorted({key[0] for key in buckets.keys()}), dtype=float)
    unique_x = np.array(sorted({key[1] for key in buckets.keys()}), dtype=float)

    heat = np.full((len(unique_theta), len(unique_x)), np.nan, dtype=float)
    theta_to_i = {value: i for i, value in enumerate(unique_theta)}
    x_to_j = {value: j for j, value in enumerate(unique_x)}

    for (t_val, x_val), tof_list in buckets.items():
        i = theta_to_i[t_val]
        j = x_to_j[x_val]
        heat[i, j] = float(np.mean(tof_list))

    return unique_theta, unique_x, heat


def plot_heatmap(theta: np.ndarray, x: np.ndarray, heat: np.ndarray, save_path: Path | None) -> None:
    plt.figure(figsize=(10, 6))

    image = plt.imshow(
        heat,
        origin="lower",
        aspect="auto",
        extent=[x.min(), x.max(), theta.min(), theta.max()],
        cmap="viridis",
    )

    plt.colorbar(image, label="Time of Flight (s)")
    plt.xlabel("x axis position (mm)")
    plt.ylabel("theta (deg)")
    plt.title("Time of Flight Heatmap")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved heatmap image to: {save_path}")

    plt.show()


def main() -> None:
    args = parse_args()

    if not args.csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv_file}")

    theta, x, tof = load_points(
        csv_path=args.csv_file,
        theta_col=args.theta_col,
        x_col=args.x_col,
        tof_col=args.tof_col,
    )
    unique_theta, unique_x, heat = build_heatmap_grid(theta=theta, x=x, tof=tof)
    plot_heatmap(theta=unique_theta, x=unique_x, heat=heat, save_path=args.save)


if __name__ == "__main__":
    main()
