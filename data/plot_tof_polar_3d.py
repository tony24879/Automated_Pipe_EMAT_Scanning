"""Plot a 3D polar (r, theta, z) Time of Flight view from a scan CSV.

Each data point uses a fixed input radius, with:
- theta (column 8, 1-based): angle around the polar axis (deg)
- z (column 9, 1-based): axis position (mm)
- color scale (column 10, 1-based): Time of Flight (s)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import Normalize

try:
    from data.plot_tof_heatmap import average_duplicate_points, load_points
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from plot_tof_heatmap import average_duplicate_points, load_points


def _path_or_none(value: str | None) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a 3D polar (r, theta, z) Time of Flight view from a CSV file.")
    parser.add_argument(
        "csv_files",
        type=Path,
        nargs="+",
        help="Path(s) to one or more CSV files (for example: data/raw/sync_scan_20260701_121153.csv)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        required=True,
        help="Fixed radius (m) used for every data point on the r axis.",
    )
    parser.add_argument(
        "--theta-col",
        type=int,
        default=8,
        help="1-based column index for theta in degrees (default: 8)",
    )
    parser.add_argument(
        "--z-col",
        type=int,
        default=9,
        help="1-based column index for the z axis position in mm (default: 9)",
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
        default="ToF 3D Polar Plot",
        help="Title text to place before the CSV filename.",
    )
    parser.add_argument(
        "--cbar-label",
        type=str,
        default="Time of Flight (s)",
        help="Label for the color bar.",
    )
    parser.add_argument(
        "--override-tof-file",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Optional CSV/text file(s) containing override TOF values, one per input CSV "
            "(in the same order)."
        ),
    )
    parser.add_argument(
        "--dual-thickness",
        action="store_true",
        help=(
            "Also plot the points in grey at the fixed radius, then plot the colored points "
            "at (radius - value) using the ToF/thickness values."
        ),
    )
    return parser.parse_args()


def default_export_path(csv_path: Path, plot_type: str) -> Path:
    plots_dir = Path(__file__).resolve().parent / "processed" / "plots"
    return plots_dir / f"{csv_path.stem}_{plot_type}_polar_3d.png"


def polar_to_cartesian(radius: float, theta_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta_rad = np.deg2rad(theta_deg)
    return radius * np.cos(theta_rad), radius * np.sin(theta_rad)


def plot_polar_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    tof: np.ndarray,
    theta: np.ndarray,
    radius: float,
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
    dual_thickness: bool = False,
) -> None:
    fig = plt.figure(figsize=(10, 7))
    axis = fig.add_subplot(111, projection="3d")

    if dual_thickness:
        # Background layer: the undeformed points at the fixed radius, with no color data.
        grey_y, grey_z = polar_to_cartesian(radius, theta)
        if plot_type == "surface":
            grey_triangulation = mtri.Triangulation(x, theta)
            axis.plot_trisurf(
                grey_triangulation,
                grey_y,
                grey_z,
                color="lightgray",
                linewidth=0.15,
                antialiased=True,
                alpha=alpha * 0.5,
            )
        else:
            axis.scatter(
                x,
                grey_y,
                grey_z,
                color="lightgray",
                s=marker_size,
                alpha=alpha * 0.5,
                depthshade=True,
            )

    if plot_type == "surface":
        # Triangulate on (axis position, theta), the natural scan grid, rather than the
        # projected cartesian coordinates, then color each face by its averaged ToF.
        triangulation = mtri.Triangulation(x, theta)
        artist = axis.plot_trisurf(
            triangulation,
            y,
            z,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            linewidth=0.15,
            antialiased=True,
        )
        face_tof = tof[triangulation.triangles].mean(axis=1)
        norm = Normalize(vmin=vmin, vmax=vmax)
        face_colors = plt.get_cmap("viridis")(norm(face_tof))
        face_colors[:, 3] = alpha
        artist.set_facecolor(face_colors)
        artist.set_array(face_tof)
    else:
        artist = axis.scatter(
            x,
            y,
            z,
            c=tof,
            cmap="viridis",
            s=marker_size,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
            depthshade=True,
        )

    axis.set_xlabel("axis position (mm)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.set_title(title)
    axis.view_init(elev=elev, azim=azim)

    axis.set_ylim(-radius, radius)
    axis.set_zlim(0, 2 * radius)

    fig.colorbar(artist, ax=axis, pad=0.1, shrink=0.75, label=cbar_label)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved 3D polar plot image to: {save_path}")

    plt.show()


def main() -> None:
    args = parse_args()
    csv_files = args.csv_files

    for csv_path in csv_files:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if args.override_tof_file is not None and len(args.override_tof_file) != len(csv_files):
        raise ValueError(
            "Number of --override-tof-file entries must match the number of input CSV files."
        )
    override_values_by_csv: list[np.ndarray | None] = []
    if args.override_tof_file is not None:
        for override_path in args.override_tof_file:
            with override_path.open("r", newline="", encoding="utf-8") as fh:
                override_values_by_csv.append(
                    np.asarray([float(line.strip()) for line in fh if line.strip()], dtype=float)
                )
    else:
        override_values_by_csv = [None] * len(csv_files)

    datasets: list[tuple[Path, np.ndarray, np.ndarray, np.ndarray]] = []
    for csv_path, override_tof_values in zip(csv_files, override_values_by_csv):
        theta, z, tof = load_points(
            csv_path=csv_path,
            theta_col=args.theta_col,
            x_col=args.z_col,
            tof_col=args.tof_col,
            override_tof_values=override_tof_values,
        )
        theta, z, tof = average_duplicate_points(theta=theta, x=z, tof=tof)
        datasets.append((csv_path, theta, z, tof))

    global_min = float(min(np.min(tof) for _, _, _, tof in datasets))
    global_max = float(max(np.max(tof) for _, _, _, tof in datasets))
    if np.isclose(global_min, global_max):
        global_max = global_min + 1e-12

    for csv_path, theta, z, tof in datasets:
        if args.save is None:
            save_path = None
        elif len(datasets) == 1 and args.save.suffix:
            save_path = args.save
        else:
            save_dir = args.save if (not args.save.suffix or args.save.is_dir()) else args.save.parent
            save_path = save_dir / f"{csv_path.stem}_{args.plot_type}_polar_3d.png"

        title_prefix = args.title_prefix.strip()
        if title_prefix:
            title = f"{title_prefix}: {csv_path.stem}"
        else:
            title = csv_path.stem

        y, z_polar = polar_to_cartesian(
            args.radius - tof if args.dual_thickness else args.radius, theta
        )
        plot_polar_3d(
            x=z,
            y=y,
            z=z_polar,
            tof=tof,
            theta=theta,
            radius=args.radius,
            plot_type=args.plot_type,
            marker_size=args.marker_size,
            alpha=args.alpha,
            elev=args.elev,
            azim=args.azim,
            save_path=save_path,
            vmin=global_min,
            vmax=global_max,
            title=title,
            cbar_label=args.cbar_label.strip(),
            dual_thickness=args.dual_thickness,
        )


if __name__ == "__main__":
    main()
