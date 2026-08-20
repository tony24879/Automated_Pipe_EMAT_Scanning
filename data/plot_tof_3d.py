"""Plot a 3D Time of Flight view from a scan CSV.

Expected default columns (1-based):
- 8: Theta (deg)
- 9: Axis Position (mm)
- 10: Time of Flight (s)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from data.plot_tof_heatmap import average_duplicate_points, load_points
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from plot_tof_heatmap import average_duplicate_points, load_points


def _path_or_none(value: str | None) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a 3D Time of Flight view from a CSV file.")
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
        "--plot-type",
        choices=["scatter", "surface"],
        default="scatter",
        help="3D plot type: scatter or triangulated surface (default: scatter)",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=12.0,
        help="Scatter marker size in points^2 (default: 12)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.9,
        help="Plot transparency from 0 to 1 (default: 0.9)",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=28.0,
        help="3D camera elevation angle in degrees (default: 28)",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=-125.0,
        help="3D camera azimuth angle in degrees (default: -125)",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="ToF 3D Plot",
        help="Title text to place before the CSV filename.",
    )
    parser.add_argument(
        "--cbar-label",
        type=str,
        default="Time of Flight (s)",
        help="Label for the color bar and z-axis.",
    )
    parser.add_argument(
        "--z-label",
        type=str,
        default=None,
        help="Optional override for the z-axis label. Defaults to the color bar label.",
    )
    parser.add_argument(
        "--override-tof-file",
        type=Path,
        default=None,
        help="Optional CSV/text file containing override TOF values for the working CSV.",
    )
    return parser.parse_args()


def default_export_path(csv_path: Path, plot_type: str) -> Path:
    plots_dir = Path(__file__).resolve().parent / "processed" / "plots"
    return plots_dir / f"{csv_path.stem}_{plot_type}_3d.png"


def plot_3d(
    x: np.ndarray,
    theta: np.ndarray,
    tof: np.ndarray,
    plot_type: str,
    marker_size: float,
    alpha: float,
    elev: float,
    azim: float,
    save_path: Path | None,
    vmin: float,
    vmax: float,
    title: str,
    cbar_label: str,
    z_label: str,
) -> None:
    fig = plt.figure(figsize=(10, 7))
    axis = fig.add_subplot(111, projection="3d")

    if plot_type == "surface":
        artist = axis.plot_trisurf(
            x,
            theta,
            tof,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            linewidth=0.15,
            antialiased=True,
            alpha=alpha,
        )
    else:
        artist = axis.scatter(
            x,
            theta,
            tof,
            c=tof,
            cmap="viridis",
            s=marker_size,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
            depthshade=True,
        )

    axis.set_xlabel("axis position (mm)")
    axis.set_ylabel("theta (deg)")
    axis.set_zlabel(z_label)
    axis.set_title(title)
    axis.view_init(elev=elev, azim=azim)

    fig.colorbar(artist, ax=axis, pad=0.1, shrink=0.75, label=cbar_label)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved 3D plot image to: {save_path}")

    plt.show()


def main() -> None:
    args = parse_args()
    csv_files = args.csv_files

    for csv_path in csv_files:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

    override_tof_values = None
    if args.override_tof_file is not None:
        if len(csv_files) > 1:
            raise ValueError("Override TOF values are only supported for a single plotted CSV.")
        with args.override_tof_file.open("r", newline="", encoding="utf-8") as fh:
            override_tof_values = np.asarray([float(line.strip()) for line in fh if line.strip()], dtype=float)

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
            save_dir = args.save if (not args.save.suffix or args.save.is_dir()) else args.save.parent
            save_path = save_dir / f"{csv_path.stem}_{args.plot_type}_3d.png"

        title_prefix = args.title_prefix.strip()
        if title_prefix:
            title = f"{title_prefix}: {csv_path.stem}"
        else:
            title = csv_path.stem

        color_label = args.z_label.strip() if args.z_label else args.cbar_label.strip()
        plot_3d(
            x=x,
            theta=theta,
            tof=tof,
            plot_type=args.plot_type,
            marker_size=args.marker_size,
            alpha=args.alpha,
            elev=args.elev,
            azim=args.azim,
            save_path=save_path,
            vmin=global_min,
            vmax=global_max,
            title=title,
            cbar_label=color_label,
            z_label=color_label,
        )


if __name__ == "__main__":
    main()