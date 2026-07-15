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
        "csv_files",
        type=Path,
        nargs="+",
        help="Path(s) to one or more CSV files (for example: data/raw/sync_scan_20260701_121153.csv)",
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
        help="Optional output image path for a single CSV input, or output directory for multiple CSV inputs.",
    )
    parser.add_argument(
        "--grid-width",
        type=int,
        default=300,
        help="Interpolated grid width in pixels (default: 300)",
    )
    parser.add_argument(
        "--grid-height",
        type=int,
        default=300,
        help="Interpolated grid height in pixels (default: 300)",
    )
    parser.add_argument(
        "--idw-power",
        type=float,
        default=2.0,
        help="Inverse-distance interpolation power (default: 2.0)",
    )
    parser.add_argument(
        "--hide-points",
        action="store_true",
        help="Hide original scatter points and only show the interpolated map.",
    )
    parser.add_argument(
        "--interpolation",
        choices=["on", "off"],
        default="on",
        help="Turn interpolation on/off (default: on)",
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


def average_duplicate_points(
    theta: np.ndarray, x: np.ndarray, tof: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Average duplicate points at identical (theta, x) coordinates.
    buckets: dict[tuple[float, float], list[float]] = defaultdict(list)
    for t_val, x_val, tof_val in zip(theta, x, tof):
        buckets[(t_val, x_val)].append(tof_val)

    theta_out: list[float] = []
    x_out: list[float] = []
    tof_out: list[float] = []
    for (t_val, x_val), tof_list in buckets.items():
        theta_out.append(t_val)
        x_out.append(x_val)
        tof_out.append(float(np.mean(tof_list)))

    return np.array(theta_out), np.array(x_out), np.array(tof_out)


def interpolate_idw(
    theta: np.ndarray,
    x: np.ndarray,
    tof: np.ndarray,
    grid_width: int,
    grid_height: int,
    power: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if grid_width < 2 or grid_height < 2:
        raise ValueError("Grid width and height must both be >= 2.")
    if power <= 0:
        raise ValueError("IDW power must be > 0.")

    x_lin = np.linspace(float(np.min(x)), float(np.max(x)), grid_width)
    theta_lin = np.linspace(float(np.min(theta)), float(np.max(theta)), grid_height)
    x_grid, theta_grid = np.meshgrid(x_lin, theta_lin)

    # Vectorized IDW interpolation over the whole grid.
    dx = x_grid[None, :, :] - x[:, None, None]
    dtheta = theta_grid[None, :, :] - theta[:, None, None]
    dist_sq = dx * dx + dtheta * dtheta

    exact_match = dist_sq == 0.0
    weights = 1.0 / np.maximum(dist_sq, 1e-12) ** (power / 2.0)
    weighted_sum = np.sum(weights * tof[:, None, None], axis=0)
    weight_total = np.sum(weights, axis=0)
    heat = weighted_sum / weight_total

    # Preserve exact sample values at coincident grid locations.
    if np.any(exact_match):
        has_exact = np.any(exact_match, axis=0)
        exact_values = np.sum(exact_match * tof[:, None, None], axis=0)
        counts = np.sum(exact_match, axis=0)
        heat[has_exact] = exact_values[has_exact] / counts[has_exact]

    return theta_grid, x_grid, heat


def default_export_path(csv_path: Path, interpolation_enabled: bool) -> Path:
    plots_dir = Path(__file__).resolve().parent / "processed" / "plots"
    suffix = "interpolated" if interpolation_enabled else "scatter"
    return plots_dir / f"{csv_path.stem}_tof_heatmap_{suffix}.png"


def plot_interpolated_heatmap(
    theta: np.ndarray,
    x: np.ndarray,
    tof: np.ndarray,
    theta_grid: np.ndarray,
    x_grid: np.ndarray,
    heat: np.ndarray,
    save_path: Path | None,
    show_points: bool,
    vmin: float,
    vmax: float,
    title: str,
) -> None:
    plt.figure(figsize=(10, 6))

    image = plt.imshow(
        heat,
        origin="lower",
        aspect="auto",
        extent=[float(np.min(x_grid)), float(np.max(x_grid)), float(np.min(theta_grid)), float(np.max(theta_grid))],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )

    if show_points:
        plt.scatter(
            x,
            theta,
            c=tof,
            cmap="viridis",
            s=12,
            edgecolors="none",
            alpha=0.8,
            vmin=vmin,
            vmax=vmax,
        )

    plt.colorbar(image, label="Error")
    plt.xlabel("x axis position (mm)")
    plt.ylabel("theta (deg)")
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved heatmap image to: {save_path}")

    plt.show()


def plot_scatter_heatmap(
    theta: np.ndarray,
    x: np.ndarray,
    tof: np.ndarray,
    save_path: Path | None,
    vmin: float,
    vmax: float,
    title: str,
) -> None:
    plt.figure(figsize=(10, 6))

    points = plt.scatter(
        x,
        theta,
        c=tof,
        cmap="viridis",
        s=18,
        edgecolors="none",
        vmin=vmin,
        vmax=vmax,
    )

    plt.colorbar(points, label="Error")
    plt.xlabel("x axis position (mm)")
    plt.ylabel("theta (deg)")
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved heatmap image to: {save_path}")

    plt.show()


def main() -> None:
    args = parse_args()
    interpolation_enabled = args.interpolation == "on"
    csv_files = args.csv_files

    for csv_path in csv_files:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

    datasets: list[tuple[Path, np.ndarray, np.ndarray, np.ndarray]] = []
    for csv_path in csv_files:
        theta, x, tof = load_points(
            csv_path=csv_path,
            theta_col=args.theta_col,
            x_col=args.x_col,
            tof_col=args.tof_col,
        )
        theta, x, tof = average_duplicate_points(theta=theta, x=x, tof=tof)
        datasets.append((csv_path, theta, x, tof))

    global_min = float(min(np.min(tof) for _, _, _, tof in datasets))
    global_max = float(max(np.max(tof) for _, _, _, tof in datasets))
    if np.isclose(global_min, global_max):
        global_max = global_min + 1e-12

    if args.save is not None and len(datasets) > 1 and args.save.suffix:
        raise ValueError(
            "When plotting multiple CSV files, --save must be a directory path (no file extension)."
        )

    for csv_path, theta, x, tof in datasets:
        if args.save is None:
            save_path = default_export_path(csv_path, interpolation_enabled)
        elif len(datasets) == 1 and args.save.suffix:
            save_path = args.save
        else:
            suffix = "interpolated" if interpolation_enabled else "scatter"
            save_path = args.save / f"{csv_path.stem}_tof_heatmap_{suffix}.png"

        title = f"ToF Error Heatmap: {csv_path.stem}"

        if interpolation_enabled:
            theta_grid, x_grid, heat = interpolate_idw(
                theta=theta,
                x=x,
                tof=tof,
                grid_width=args.grid_width,
                grid_height=args.grid_height,
                power=args.idw_power,
            )
            plot_interpolated_heatmap(
                theta=theta,
                x=x,
                tof=tof,
                theta_grid=theta_grid,
                x_grid=x_grid,
                heat=heat,
                save_path=save_path,
                show_points=not args.hide_points,
                vmin=global_min,
                vmax=global_max,
                title=title,
            )
        else:
            plot_scatter_heatmap(
                theta=theta,
                x=x,
                tof=tof,
                save_path=save_path,
                vmin=global_min,
                vmax=global_max,
                title=title,
            )


if __name__ == "__main__":
    main()
