"""Plan and execute EMAT scans around a horizontal cylinder with serpentine theta sweeps."""

import argparse
import json
import math
import time
from pathlib import Path

from config.robot_config import ROBOT_IP
from emat.emat_interface import EMATSession
from emat.live_plot import LiveWaveformPlot
from emat.sync_logger import SyncLogger
from robot.connection import RobotConnection
from robot.lite6 import Lite6
from robot.setup import RobotSetup


class HorizontalCylindricalScanPlanner:
    """Generate horizontal-cylinder waypoints using an outer transition ring."""

    def __init__(
        self,
        centre,
        radius,
        length,
        lift_off,
        outer_offset_mm=10.0,
        x_start=None,
        x_end=None,
        theta_limit_a_deg=0.0,
        theta_limit_b_deg=180.0,
        scan_points_per_x=12,
        x_scans=5,
    ):
        self.centre = [float(c) for c in centre]
        self.radius = float(radius)
        self.length = float(length)
        self.lift_off = float(lift_off)
        self.outer_offset_mm = max(0.0, float(outer_offset_mm))

        default_x_start = self.centre[0]
        default_x_end = self.centre[0] + self.length
        self.x_start = float(default_x_start if x_start is None else x_start)
        self.x_end = float(default_x_end if x_end is None else x_end)

        self.theta_limit_a_deg = float(theta_limit_a_deg)
        self.theta_limit_b_deg = float(theta_limit_b_deg)

        self.scan_points_per_x = max(2, int(scan_points_per_x))
        self.x_scans = max(1, int(x_scans))

    def _x_positions(self):
        """Compute axis-step positions along cylinder length."""
        if self.x_scans <= 1:
            return [self.x_start]
        return [
            self.x_start + (self.x_end - self.x_start) * i / (self.x_scans - 1)
            for i in range(self.x_scans)
        ]

    def _closest_equivalent_angle(self, target_deg, reference_deg):
        """Choose equivalent angle closest to reference for smooth wrist commands."""
        candidates = [target_deg - 720.0, target_deg - 360.0, target_deg, target_deg + 360.0, target_deg + 720.0]
        return min(candidates, key=lambda c: (abs(c - reference_deg), abs(c)))

    def _unwrap_near(self, angle_deg, reference_deg):
        """Unwrap an angle so it stays close to a reference angle."""
        if reference_deg is None:
            return angle_deg
        return self._closest_equivalent_angle(angle_deg, reference_deg)

    def _angles_for_sweep(self, start_deg, end_deg, count):
        """Generate evenly spaced angular samples from start to end."""
        if count <= 1:
            return [start_deg]
        return [start_deg + (end_deg - start_deg) * i / (count - 1) for i in range(count)]

    def _roll_from_theta(self, theta_deg, previous_roll):
        """Map theta to roll for radial tool normal alignment around X-axis."""
        base_roll = theta_deg + 90.0
        if previous_roll is None:
            return base_roll
        return self._closest_equivalent_angle(base_roll, previous_roll)

    def _append_point(self, points, radius_mm, theta_deg, x_mm, previous_roll, capture):
        """Append one Cartesian waypoint from horizontal-cylinder polar coordinates."""
        theta = math.radians(theta_deg)
        y = self.centre[1] + radius_mm * math.cos(theta)
        z = self.centre[2] + radius_mm * math.sin(theta)

        roll = self._roll_from_theta(theta_deg, previous_roll)
        pitch = 0.0
        yaw = 0.0

        points.append((x_mm, y, z, roll, pitch, yaw, capture))
        return roll

    def generate(self):
        """Build serpentine theta sweeps with outer-ring transitions between all inner points."""
        points = []
        r_scan = self.radius + self.lift_off
        r_outer = r_scan + self.outer_offset_mm

        x_positions = self._x_positions()
        if not x_positions:
            return points

        theta_a = self.theta_limit_a_deg
        theta_b = self._unwrap_near(self.theta_limit_b_deg, theta_a)

        previous_roll = None

        # Radial standoff approach before first contact.
        approach_r = r_outer + 50.0
        first_x = x_positions[0]
        first_theta = theta_a
        previous_roll = self._append_point(points, approach_r, first_theta, first_x, previous_roll, capture=False)

        def move_inner_to_inner(theta_from, theta_to, x_mm, prev_roll, capture_to):
            """Move between adjacent inner points using the outer ring as a corridor."""
            prev_roll = self._append_point(points, r_outer, theta_from, x_mm, prev_roll, capture=False)
            prev_roll = self._append_point(points, r_outer, theta_to, x_mm, prev_roll, capture=False)
            prev_roll = self._append_point(points, r_scan, theta_to, x_mm, prev_roll, capture=capture_to)
            return prev_roll

        # Enter first layer at theta_a.
        previous_roll = self._append_point(points, r_outer, theta_a, first_x, previous_roll, capture=False)
        previous_roll = self._append_point(points, r_scan, theta_a, first_x, previous_roll, capture=True)

        prev_x = first_x
        for layer_index, x_mm in enumerate(x_positions):
            if layer_index % 2 == 0:
                sweep_start = theta_a
                sweep_end = theta_b
            else:
                sweep_start = theta_b
                sweep_end = theta_a

            sweep_thetas = self._angles_for_sweep(sweep_start, sweep_end, self.scan_points_per_x)

            if layer_index > 0:
                # Transition along axis from previous sweep endpoint while staying on the same theta.
                previous_roll = self._append_point(points, r_outer, sweep_start, prev_x, previous_roll, capture=False)
                previous_roll = self._append_point(points, r_outer, sweep_start, x_mm, previous_roll, capture=False)
                previous_roll = self._append_point(points, r_scan, sweep_start, x_mm, previous_roll, capture=True)

            for i in range(len(sweep_thetas) - 1):
                previous_roll = move_inner_to_inner(
                    sweep_thetas[i],
                    sweep_thetas[i + 1],
                    x_mm,
                    previous_roll,
                    capture_to=True,
                )

            prev_x = x_mm

        return points

    def save(self, filename):
        """Save planner settings to JSON."""
        payload = {
            "centre": self.centre,
            "radius": self.radius,
            "length": self.length,
            "lift_off": self.lift_off,
            "outer_offset_mm": self.outer_offset_mm,
            "x_start": self.x_start,
            "x_end": self.x_end,
            "theta_limit_a_deg": self.theta_limit_a_deg,
            "theta_limit_b_deg": self.theta_limit_b_deg,
            "scan_points_per_x": self.scan_points_per_x,
            "x_scans": self.x_scans,
        }
        Path(filename).write_text(json.dumps(payload, indent=2))


def _resolve_calibration_file(calibration_file):
    """Resolve calibration file path from common user/project locations."""
    if not calibration_file:
        return None

    candidates = []
    provided = Path(calibration_file)
    candidates.append(provided)
    candidates.append(Path.cwd() / provided)
    root = Path(__file__).resolve().parent.parent
    candidates.append(root / provided)
    candidates.append(root / "data" / "raw" / provided.name)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(provided)


def _read_horizontal_calibration_geometry(calibration_file):
    """Read horizontal-cylinder scan geometry from calibration JSON."""
    resolved_file = _resolve_calibration_file(calibration_file)
    if not resolved_file:
        return None

    try:
        payload = json.loads(Path(resolved_file).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    required = [
        "centre",
        "radius",
        "x_start",
        "x_end",
        "theta_limit_a_deg",
        "theta_limit_b_deg",
    ]
    if any(key not in payload for key in required):
        return None

    centre = [float(v) for v in payload["centre"]]
    radius = float(payload["radius"])
    x_start = float(payload["x_start"])
    x_end = float(payload["x_end"])
    theta_a = float(payload["theta_limit_a_deg"])
    theta_b = float(payload["theta_limit_b_deg"])

    return {
        "resolved_file": resolved_file,
        "centre": centre,
        "radius": radius,
        "length": abs(x_end - x_start),
        "x_start": x_start,
        "x_end": x_end,
        "theta_limit_a_deg": theta_a,
        "theta_limit_b_deg": theta_b,
    }


def run_horizontal_cylindrical_scan(
    centre,
    radius,
    length,
    lift_off,
    outer_offset_mm=10.0,
    dwell_seconds=5.0,
    speed=40,
    output_folder="data/raw",
    calibration_file=None,
    scan_points_per_x=12,
    x_scans=5,
    theta_limit_a_deg=0.0,
    theta_limit_b_deg=180.0,
):
    """Execute horizontal cylindrical scan and log synchronized EMAT + robot data."""
    calibration = _read_horizontal_calibration_geometry(calibration_file)
    if calibration is not None:
        centre = calibration["centre"]
        radius = calibration["radius"]
        length = calibration["length"]
        x_start = calibration["x_start"]
        x_end = calibration["x_end"]
        theta_limit_a_deg = calibration["theta_limit_a_deg"]
        theta_limit_b_deg = calibration["theta_limit_b_deg"]

        print(f"Using horizontal geometry from {calibration['resolved_file']}")
        print(f"Calibrated centre: x={centre[0]:.1f}, y={centre[1]:.1f}, z={centre[2]:.1f}")
        print(f"Calibrated radius: {radius:.1f} mm")
        print(f"Calibrated x range: {x_start:.1f} mm to {x_end:.1f} mm")
        print(f"Theta limits: {theta_limit_a_deg:.1f} deg -> {theta_limit_b_deg:.1f} deg")
    else:
        x_start = centre[0]
        x_end = centre[0] + length
        print("No valid horizontal calibration file found; using CLI/default geometry")

    planner = HorizontalCylindricalScanPlanner(
        centre=centre,
        radius=radius,
        length=length,
        lift_off=lift_off,
        outer_offset_mm=outer_offset_mm,
        x_start=x_start,
        x_end=x_end,
        theta_limit_a_deg=theta_limit_a_deg,
        theta_limit_b_deg=theta_limit_b_deg,
        scan_points_per_x=scan_points_per_x,
        x_scans=x_scans,
    )

    points = planner.generate()

    conn = RobotConnection(ROBOT_IP)
    arm = conn.connect()
    robot = Lite6(arm)
    setup = RobotSetup(arm)
    setup.configure()

    plotter = LiveWaveformPlot()
    logger = SyncLogger(folder=output_folder)

    try:
        with EMATSession() as emat:
            print("Configuring EMAT...")
            emat.configure()
            execution_points = list(points)

            # Build a safer startup transit: Z lift, XY move at clearance, then continue.
            if points:
                try:
                    current_pose = robot.get_pose()
                    if current_pose and len(current_pose) >= 6:
                        current_x = float(current_pose[0])
                        current_y = float(current_pose[1])
                        current_z = float(current_pose[2])
                        current_roll = float(current_pose[3])
                        current_pitch = float(current_pose[4])
                        current_yaw = float(current_pose[5])
                        first_x, first_y, first_z, first_roll, first_pitch, first_yaw, _ = points[0]

                        startup_clearance_mm = 80.0
                        safe_z = max(current_z, first_z + startup_clearance_mm)

                        z_lift_waypoint = (current_x, current_y, safe_z, current_roll, current_pitch, current_yaw, False)
                        xy_transit_waypoint = (first_x, first_y, safe_z, current_roll, current_pitch, current_yaw, False)
                        rotate_at_clearance_waypoint = (first_x, first_y, safe_z, first_roll, first_pitch, first_yaw, False)

                        execution_points = [z_lift_waypoint, xy_transit_waypoint, rotate_at_clearance_waypoint] + points
                        print(
                            "Safer startup transit enabled: "
                            f"lift to z={safe_z:.1f} mm, XY at clearance, then rotate before descent"
                        )
                except Exception as exc:
                    print(f"Warning: unable to create split first move ({exc}); using original path")

            capture_count = sum(1 for *_, capture in execution_points if capture)
            reset_count = len(execution_points) - capture_count

            print(f"Starting horizontal cylindrical scan with {len(execution_points)} motion points")
            print(f"Capture points: {capture_count}, reset/approach points: {reset_count}")
            print(f"X range: {x_start:.1f} mm to {x_end:.1f} mm")
            print(f"Using {scan_points_per_x} scan points per x layer and {x_scans} x layers")

            for index, (x, y, z, roll, pitch, yaw, capture) in enumerate(execution_points, start=1):
                action = "SCAN" if capture else "RESET"
                print(
                    f"[{index}/{len(execution_points)}] {action} move to "
                    f"x={x:.1f}, y={y:.1f}, z={z:.1f}, roll={roll:.1f}, pitch={pitch:.1f}, yaw={yaw:.1f}"
                )
                move_code = robot.move_to(x, y, z, speed=speed, roll=roll, pitch=pitch, yaw=yaw)
                if move_code == 9:
                    arm.set_state(0)
                    time.sleep(0.1)
                    move_code = robot.move_to(x, y, z, speed=speed, roll=roll, pitch=pitch, yaw=yaw)

                if move_code != 0:
                    state_code, state = arm.get_state()
                    err_code, err_warn = arm.get_err_warn_code(show=True)
                    raise RuntimeError(
                        "Move failed at point "
                        f"{index}/{len(execution_points)} with API code={move_code}; "
                        f"state_query_code={state_code}, state={state}, "
                        f"err_query_code={err_code}, err_warn={err_warn}"
                    )

                if capture:
                    dwell_until = time.monotonic() + dwell_seconds
                    data = None
                    while time.monotonic() < dwell_until:
                        data = emat.acquire()
                        plotter.update(data)
                        time.sleep(0.1)

                    pose = robot.get_pose()
                    logger.log(pose, data)
                    print(f"Captured point {index}/{len(execution_points)}")

            print("Horizontal cylindrical scan complete")
    finally:
        plotter.close()
        logger.close()
        conn.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a cylindrical EMAT scan around a horizontal cylinder")
    parser.add_argument("--centre", type=float, nargs=3, default=[250.0, 0.0, 150.0], help="Cylinder centre x y z")
    parser.add_argument("--radius", type=float, default=50.0, help="Cylinder radius in mm")
    parser.add_argument("--length", type=float, default=150.0, help="Cylinder scan length along x in mm")
    parser.add_argument("--lift-off", type=float, default=1.0, help="Radial lift-off from cylinder surface in mm")
    parser.add_argument("--outer-offset-mm", type=float, default=10.0, help="Extra radial offset for transition ring in mm")
    parser.add_argument("--theta-limit-a-deg", type=float, default=0.0, help="First angular limit in yz-plane degrees")
    parser.add_argument("--theta-limit-b-deg", type=float, default=180.0, help="Second angular limit in yz-plane degrees")
    parser.add_argument("--dwell", type=float, default=5.0, help="Seconds to dwell at each point")
    parser.add_argument("--speed", type=float, default=40.0, help="Motion speed")
    parser.add_argument("--calibration-file", type=str, default=None, help="Optional saved calibration JSON file")
    parser.add_argument("--output-folder", type=str, default="data/raw", help="Folder for scan logs")
    parser.add_argument("--scan-points-per-x", type=int, default=None, help="Number of scan points per x layer")
    parser.add_argument("--x-scans", type=int, default=None, help="Number of x layers to scan")
    args = parser.parse_args()

    scan_points_per_x = args.scan_points_per_x
    if scan_points_per_x is None:
        scan_points_per_x = int(input("Number of scan points per x layer: ") or 12)

    x_scans = args.x_scans
    if x_scans is None:
        x_scans = int(input("Number of x layers to scan: ") or 5)

    run_horizontal_cylindrical_scan(
        centre=args.centre,
        radius=args.radius,
        length=args.length,
        lift_off=args.lift_off,
        outer_offset_mm=args.outer_offset_mm,
        dwell_seconds=args.dwell,
        speed=args.speed,
        output_folder=args.output_folder,
        calibration_file=args.calibration_file,
        scan_points_per_x=scan_points_per_x,
        x_scans=x_scans,
        theta_limit_a_deg=args.theta_limit_a_deg,
        theta_limit_b_deg=args.theta_limit_b_deg,
    )
