"""Interactive calibration for horizontal-cylinder scans with theta-limit teaching."""

import json
import math
from pathlib import Path

from config.robot_config import ROBOT_IP
from robot.connection import RobotConnection
from robot.lite6 import Lite6
from robot.setup import RobotSetup


def _fit_circle_2d(points):
    """Least-squares algebraic circle fit to a list of 2D points."""
    n = len(points)
    if n < 3:
        raise ValueError("At least 3 points required for a circle fit")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    r2s = [x ** 2 + y ** 2 for x, y in zip(xs, ys)]

    sxx = sum(x ** 2 for x in xs)
    syy = sum(y ** 2 for y in ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sx = sum(xs)
    sy = sum(ys)
    sxr = sum(x * r for x, r in zip(xs, r2s))
    syr = sum(y * r for y, r in zip(ys, r2s))
    sr = sum(r2s)

    mat = [
        [sxx, sxy, sx, -sxr],
        [sxy, syy, sy, -syr],
        [sx, sy, n, -sr],
    ]

    for col in range(3):
        max_row = max(range(col, 3), key=lambda r: abs(mat[r][col]))
        mat[col], mat[max_row] = mat[max_row], mat[col]
        if abs(mat[col][col]) < 1e-12:
            raise ValueError("Circle fit failed - points may be collinear or too close together")
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


def _closest_equivalent_angle(target_deg, reference_deg):
    """Choose equivalent target angle nearest to reference angle."""
    candidates = [target_deg - 720.0, target_deg - 360.0, target_deg, target_deg + 360.0, target_deg + 720.0]
    return min(candidates, key=lambda c: (abs(c - reference_deg), abs(c)))


def _theta_yz_deg(point, cy, cz):
    """Compute yz-plane polar angle (deg) for a point around horizontal cylinder axis."""
    return math.degrees(math.atan2(point[2] - cz, point[1] - cy))


def save_calibration(filename, centre, radius, x_start, x_end, theta_limit_a_deg, theta_limit_b_deg, raw_points):
    """Persist fitted horizontal-cylinder geometry and taught points to JSON."""
    calibration = {
        "centre": [float(v) for v in centre],
        "radius": float(radius),
        "x_start": float(x_start),
        "x_end": float(x_end),
        "theta_limit_a_deg": float(theta_limit_a_deg),
        "theta_limit_b_deg": float(theta_limit_b_deg),
        "raw_points": [[float(v) for v in p] for p in raw_points],
    }
    Path(filename).write_text(json.dumps(calibration, indent=2))


def main():
    """Guide operator through horizontal-cylinder calibration and save geometry."""
    calibration_file = Path("data/raw/cylinder_calibration_horizontal.json")
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

        print("\n--- PHASE 1: Circumferential circle fit (YZ plane) ---")
        print("Touch 4 surface points around the circumference at approximately the SAME x.")
        print("Important: point 1 and point 2 will become your scan theta limits.")
        print("For top-surface-only scans, keep these two limits within ~180 degrees span.")

        circ_points = []
        for i in range(4):
            input(f"\nCircumference point {i + 1}/4 - move to surface, then press ENTER")
            pose = robot.get_pose()
            circ_points.append(pose)
            print(f"  Recorded: x={pose[0]:.2f}  y={pose[1]:.2f}  z={pose[2]:.2f}")

        print("\n--- PHASE 2: Axis range (X limits) ---")
        input("\nMove to one end of the scan range along cylinder axis (x-start), then press ENTER")
        x_start_pose = robot.get_pose()
        print(f"  Axis start x: {x_start_pose[0]:.2f} mm")

        input("\nMove to the opposite end of the scan range along cylinder axis (x-end), then press ENTER")
        x_end_pose = robot.get_pose()
        print(f"  Axis end x: {x_end_pose[0]:.2f} mm")

        print("Restoring normal robot mode...")
        arm.set_mode(0)
        arm.set_state(0)
    finally:
        conn.disconnect()

    yz_points = [(p[1], p[2]) for p in circ_points]
    cy, cz, radius = _fit_circle_2d(yz_points)

    x_start = min(float(x_start_pose[0]), float(x_end_pose[0]))
    x_end = max(float(x_start_pose[0]), float(x_end_pose[0]))
    centre_x = 0.5 * (x_start + x_end)
    centre = [centre_x, cy, cz]

    theta_a = _theta_yz_deg(circ_points[0], cy, cz)
    theta_b_raw = _theta_yz_deg(circ_points[1], cy, cz)
    theta_b = _closest_equivalent_angle(theta_b_raw, theta_a)
    theta_span = abs(theta_b - theta_a)

    print(f"\nFitted circle centre: y={cy:.2f}  z={cz:.2f}")
    print(f"Fitted radius:        {radius:.2f} mm")
    print(f"Axis range (x):       {x_start:.2f} mm  ->  {x_end:.2f} mm")
    print(f"Theta limits:         {theta_a:.2f} deg  ->  {theta_b:.2f} deg")
    print(f"Theta span:           {theta_span:.2f} deg")

    if theta_span > 180.0 + 1e-6:
        print("Warning: taught theta span exceeds 180 degrees; top-surface scan intent may be violated.")

    print("\nPer-point residuals in YZ circle fit (ideal: +/-1 mm):")
    for i, (py, pz) in enumerate(yz_points):
        dist = math.sqrt((py - cy) ** 2 + (pz - cz) ** 2)
        print(f"  Point {i + 1}: {dist - radius:+.2f} mm")

    all_points = circ_points + [x_start_pose, x_end_pose]
    save_calibration(
        calibration_file,
        centre,
        radius,
        x_start,
        x_end,
        theta_a,
        theta_b,
        all_points,
    )
    print(f"\nHorizontal calibration saved to {calibration_file}")


if __name__ == "__main__":
    main()
