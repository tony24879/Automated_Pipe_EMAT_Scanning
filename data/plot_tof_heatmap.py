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
from scipy.interpolate import griddata


def _path_or_none(value: str | None) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value)


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
        type=_path_or_none,
        default="",
        help="Optional output image path. If blank or omitted, no image is saved.",
    )
    parser.add_argument(
        "--grid-width",
        type=int,
        default=500,
        help="Interpolated grid width in pixels (default: 500)",
    )
    parser.add_argument(
        "--grid-height",
        type=int,
        default=500,
        help="Interpolated grid height in pixels (default: 500)",
    )
    parser.add_argument(
        "--idw-power",
        type=float,
        default=2.0,
        help="Inverse-distance interpolation power (default: 2.0)",
    )
    parser.add_argument(
        "--interpolation-method",
        choices=["cubic", "linear", "idw"],
        default="cubic",
        help="Interpolation method used to build the heatmap surface (default: cubic)",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=0.8,
        help="Gaussian smoothing sigma applied to the interpolated grid in pixels (default: 0.8)",
    )
    parser.add_argument(
        "--show-points",
        action="store_true",
        help="Overlay original scatter points on top of the interpolated map.",
    )
    parser.add_argument(
        "--hide-points",
        action="store_true",
        help="Deprecated alias for compatibility; interpolated plots hide points by default.",
    )
    parser.add_argument(
        "--override-tof-file",
        type=Path,
        default=None,
        help="Optional CSV/text file containing override TOF values for the working CSV.",
    )
    parser.add_argument(
        "--interpolation",
        choices=["on", "off"],
        default="on",
        help="Turn interpolation on/off (default: on)",
    )
    parser.add_argument(
        "--image-interpolation",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        default="bicubic",
        help="Matplotlib image resampling mode for the interpolated heatmap (default: bicubic)",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="ToF Error Heatmap",
        help="Title text to place before the CSV filename.",
    )
    parser.add_argument(
        "--cbar-label",
        type=str,
        default="ToF Error",
        help="Label for the color bar.",
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


def load_points(
    csv_path: Path,
    theta_col: int,
    x_col: int,
    tof_col: int,
    override_tof_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            raw_theta = row[theta_idx].strip()
            raw_x = row[x_idx].strip()
            raw_tof = row[tof_idx].strip()
            if not raw_theta or not raw_x or not raw_tof:
                continue
            try:
                theta = float(raw_theta)
                x = float(raw_x)
                tof = float(raw_tof)
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

    if override_tof_values is not None:
        if len(override_tof_values) != len(tof_vals):
            raise ValueError(
                "Override TOF values length does not match the number of rows in the source CSV. "
                f"Expected {len(tof_vals)}, got {len(override_tof_values)}."
            )
        tof_vals = [float(value) for value in override_tof_values]

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


def interpolate_griddata(
    theta: np.ndarray,
    x: np.ndarray,
    tof: np.ndarray,
    grid_width: int,
    grid_height: int,
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if grid_width < 2 or grid_height < 2:
        raise ValueError("Grid width and height must both be >= 2.")

    x_lin = np.linspace(float(np.min(x)), float(np.max(x)), grid_width)
    theta_lin = np.linspace(float(np.min(theta)), float(np.max(theta)), grid_height)
    x_grid, theta_grid = np.meshgrid(x_lin, theta_lin)

    points = np.column_stack((x, theta))
    heat = griddata(points, tof, (x_grid, theta_grid), method=method)

    # Cubic/linear can be undefined near convex-hull edges; fill holes with
    # nearest-neighbor values to keep exported images fully populated.
    if np.isnan(heat).any():
        nearest = griddata(points, tof, (x_grid, theta_grid), method="nearest")
        heat = np.where(np.isnan(heat), nearest, heat)

    return theta_grid, x_grid, np.asarray(heat, dtype=float)


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= np.sum(kernel)
    return kernel


def _convolve_along_axis(array: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = len(kernel) // 2
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(array, pad_width, mode="edge")
    return np.apply_along_axis(lambda values: np.convolve(values, kernel, mode="valid"), axis, padded)


def smooth_heatmap(heat: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return heat

    kernel = _gaussian_kernel1d(sigma)
    smoothed = _convolve_along_axis(heat, kernel, axis=0)
    smoothed = _convolve_along_axis(smoothed, kernel, axis=1)
    return smoothed


def default_export_path(csv_path: Path, interpolation_enabled: bool) -> Path:
    plots_dir = Path(__file__).resolve().parent / "processed" / "plots"
    suffix = "heatmap" if not interpolation_enabled else "heatmap_interpolated"
    return plots_dir / f"{csv_path.stem}_{suffix}.png"


def plot_interpolated_heatmap(
    theta: np.ndarray,
    x: np.ndarray,
    tof: np.ndarray,
    theta_grid: np.ndarray,
    x_grid: np.ndarray,
    heat: np.ndarray,
    save_path: Path | None,
    show_points: bool,
    image_interpolation: str,
    vmin: float,
    vmax: float,
    title: str,
    cbar_label: str,
) -> None:
    plt.figure(figsize=(10, 6))

    image = plt.imshow(
        heat,
        origin="lower",
        aspect="auto",
        extent=[float(np.min(x_grid)), float(np.max(x_grid)), float(np.min(theta_grid)), float(np.max(theta_grid))],
        cmap="viridis",
        interpolation=image_interpolation,
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

    plt.colorbar(image, label=cbar_label)
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
    cbar_label: str,
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

    plt.colorbar(points, label=cbar_label)
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

    override_tof_values = load_override_tof_values(args.override_tof_file) if args.override_tof_file is not None else None
    datasets: list[tuple[Path, np.ndarray, np.ndarray, np.ndarray]] = []
    for csv_path in csv_files:
        theta, x, tof = load_points(
            csv_path=csv_path,
            theta_col=args.theta_col,
            x_col=args.x_col,
            tof_col=args.tof_col,
            override_tof_values=override_tof_values,
        )
        theta, x, tof = average_duplicate_points(theta=theta, x=x, tof=tof)
        datasets.append((csv_path, theta, x, tof))

    # Use one shared color scale across all inputs so figure-to-figure
    # comparisons reflect signal differences rather than autoscaling.
    global_min = float(min(np.min(tof) for _, _, _, tof in datasets))
    global_max = float(max(np.max(tof) for _, _, _, tof in datasets))
    if np.isclose(global_min, global_max):
        global_max = global_min + 1e-12

    for csv_path, theta, x, tof in datasets:
        if args.save is None:
            save_path = None
        elif len(datasets) == 1 and args.save.suffix:
            save_path = args.save
        else:
            suffix = "heatmap_interpolated" if interpolation_enabled else "heatmap"
            save_dir = args.save if (not args.save.suffix or args.save.is_dir()) else args.save.parent
            save_path = save_dir / f"{csv_path.stem}_{suffix}.png"

        title_prefix = args.title_prefix.strip()
        if title_prefix:
            title = f"{title_prefix}: {csv_path.stem}"
        else:
            title = csv_path.stem

        if interpolation_enabled:
            # IDW is robust for sparse data; griddata methods are smoother but
            # may extrapolate less predictably near boundaries.
            if args.interpolation_method == "idw":
                theta_grid, x_grid, heat = interpolate_idw(
                    theta=theta,
                    x=x,
                    tof=tof,
                    grid_width=args.grid_width,
                    grid_height=args.grid_height,
                    power=args.idw_power,
                )
            else:
                theta_grid, x_grid, heat = interpolate_griddata(
                    theta=theta,
                    x=x,
                    tof=tof,
                    grid_width=args.grid_width,
                    grid_height=args.grid_height,
                    method=args.interpolation_method,
                )
            heat = smooth_heatmap(heat=heat, sigma=args.smooth_sigma)
            plot_interpolated_heatmap(
                theta=theta,
                x=x,
                tof=tof,
                theta_grid=theta_grid,
                x_grid=x_grid,
                heat=heat,
                save_path=save_path,
                show_points=args.show_points and not args.hide_points,
                image_interpolation=args.image_interpolation,
                vmin=global_min,
                vmax=global_max,
                title=title,
                cbar_label=args.cbar_label,
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
                cbar_label=args.cbar_label,
            )


if __name__ == "__main__":
    main()
