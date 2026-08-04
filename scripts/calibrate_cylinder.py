"""Interactive cylinder calibration via circumferential touch points and least-squares circle fit."""

import json
import math
from pathlib import Path

from config.robot_config import ROBOT_IP
from robot.connection import RobotConnection
from robot.lite6 import Lite6
from robot.setup import RobotSetup


def _fit_circle_2d(points):
    """Least-squares algebraic circle fit to a list of (x, y) points.

    Solves the overdetermined linear system derived from
      (x - cx)^2 + (y - cy)^2 = r^2
    which linearises to:
      A*x + B*y + C = -(x^2 + y^2)   with cx=-A/2, cy=-B/2, r=sqrt(cx^2+cy^2-C)

    Returns (cx, cy, radius). Requires at least 3 non-collinear points.
    """
    n = len(points)
    if n < 3:
        raise ValueError("At least 3 points required for a circle fit")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    r2s = [x ** 2 + y ** 2 for x, y in zip(xs, ys)]

    sxx = sum(x ** 2 for x in xs)
    syy = sum(y ** 2 for y in ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sx  = sum(xs)
    sy  = sum(ys)
    sxr = sum(x * r for x, r in zip(xs, r2s))
    syr = sum(y * r for y, r in zip(ys, r2s))
    sr  = sum(r2s)

    mat = [
        [sxx, sxy, sx, -sxr],
        [sxy, syy, sy, -syr],
        [sx,  sy,  n,  -sr ],
    ]

    for col in range(3):
        max_row = max(range(col, 3), key=lambda r: abs(mat[r][col]))
        mat[col], mat[max_row] = mat[max_row], mat[col]
        if abs(mat[col][col]) < 1e-12:
            raise ValueError(
                "Circle fit failed — points may be collinear or too close together"
            )
        for row in range(col + 1, 3):
            factor = mat[row][col] / mat[col][col]
            for k in range(col, 4):
                mat[row][k] -= factor * mat[col][k]

    sol = [0.0, 0.0, 0.0]
    for row in range(2, -1, -1):
        sol[row] = mat[row][3]
        for k in range(row + 1, 3):
            sol[row] -= mat[row][k] * sol[k]
        sol[row] /= mat[row][row]

    a_coeff, b_coeff, c_coeff = sol
    cx = -a_coeff / 2.0
    cy = -b_coeff / 2.0
    radius = math.sqrt(max(0.0, cx ** 2 + cy ** 2 - c_coeff))
    return cx, cy, radius


def save_calibration(filename, centre, radius, z_start, z_end, raw_points):
    """Persist fitted calibration geometry and raw touch points to JSON."""
    calibration = {
        "centre": [float(v) for v in centre],
        "radius": float(radius),
        "z_start": float(z_start),
        "z_end": float(z_end),
        "raw_points": [[float(v) for v in p] for p in raw_points],
    }
    Path(filename).write_text(json.dumps(calibration, indent=2))


def main():
    """Guide operator through circumferential touch teaching and save calibration.

    Teaching protocol:
      Phase 1 - four surface-contact points spaced ~90 degrees apart at the SAME height
                 (use the equator/mid-height of the cylinder for best access).
                 These are used for least-squares circle fitting -> centre XY + radius.
      Phase 2 - one point at the cylinder BASE and one at the cylinder TOP.
                 Only the Z coordinate of each is used -> z_start / z_end.

    Quality check: residuals for each phase-1 point are printed after fitting.
    A residual of +/-1 mm or less indicates a good calibration.
    """
    calibration_file = Path("data/raw/cylinder_calibration.json")
    calibration_file.parent.mkdir(parents=True, exist_ok=True)

    conn = RobotConnection(ROBOT_IP)
    arm = conn.connect()
    robot = Lite6(arm)
    setup = RobotSetup(arm)
    setup.configure()

    try:
        print("Switching robot to teach mode for manual touch movement...")
        arm.motion_enable(True)
        arm.set_mode(2)
        arm.set_state(0)

        print("\n--- PHASE 1: Circumferential circle fit ---")
        print("Touch the EMAT to the cylinder surface at 4 positions around the")
        print("circumference, all at approximately the SAME height (mid-cylinder).")
        print("Space them roughly 90 degrees apart (think N / E / S / W).")
        circ_points = []
        for i in range(4):
            input(f"\nCircumferential point {i + 1}/4 — move to surface, then press ENTER")
            pose = robot.get_pose()
            circ_points.append(pose)
            print(f"  Recorded: x={pose[0]:.2f}  y={pose[1]:.2f}  z={pose[2]:.2f}")

        print("\n--- PHASE 2: Z range ---")
        input("\nMove to the BOTTOM surface contact (cylinder base), then press ENTER")
        z_bottom_pose = robot.get_pose()
        print(f"  Bottom z: {z_bottom_pose[2]:.2f} mm")

        input("\nMove to the TOP surface contact (cylinder top), then press ENTER")
        z_top_pose = robot.get_pose()
        print(f"  Top z: {z_top_pose[2]:.2f} mm")

        print("Restoring normal robot mode...")
        arm.set_mode(0)
        arm.set_state(0)
    finally:
        conn.disconnect()

    # Fit circle to the 4 circumferential TCP positions.
    xy_points = [(p[0], p[1]) for p in circ_points]
    cx, cy, radius = _fit_circle_2d(xy_points)
    z_equator = sum(p[2] for p in circ_points) / len(circ_points)
    centre = [cx, cy, z_equator]

    z_start = min(z_bottom_pose[2], z_top_pose[2])
    z_end   = max(z_bottom_pose[2], z_top_pose[2])

    print(f"\nFitted circle centre: x={cx:.2f}  y={cy:.2f}")
    print(f"Fitted radius:        {radius:.2f} mm")
    print(f"Z range:              {z_start:.2f} mm  ->  {z_end:.2f} mm")
    print("\nPer-point residuals (ideal: +/-1 mm):")
    for i, (px, py) in enumerate(xy_points):
        dist = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        print(f"  Point {i + 1}: {dist - radius:+.2f} mm")

    all_points = circ_points + [z_bottom_pose, z_top_pose]
    save_calibration(calibration_file, centre, radius, z_start, z_end, all_points)
    print(f"\nCalibration saved to {calibration_file}")


if __name__ == "__main__":
    main()
