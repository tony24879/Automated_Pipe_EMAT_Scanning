"""Plan and execute cylindrical EMAT scans with wrist-safe angular sequencing."""

import argparse
import json
import math
import time
from pathlib import Path

from robot.connection import RobotConnection
from robot.lite6 import Lite6
from robot.setup import RobotSetup
from emat.emat_interface import EMATSession
from emat.live_plot import LiveWaveformPlot
from emat.sync_logger import SyncLogger
from config.robot_config import ROBOT_IP


class CylindricalScanPlanner:
    """Generate wrist-safe cylindrical scan waypoints across Z layers."""

    def __init__(self, centre, radius, height, lift_off, theta_step_deg, z_step, z_start=None, z_end=None, scan_points_per_z=12, z_scans=5, theta_start_deg=180.0):
        """Store scan geometry and sampling parameters."""
        self.centre = [float(c) for c in centre]
        self.radius = float(radius)
        self.height = float(height)
        self.lift_off = float(lift_off)
        self.theta_step_deg = float(theta_step_deg)
        self.z_step = float(z_step)
        self.z_start = float(z_start) if z_start is not None else self.centre[2]
        self.z_end = float(z_end) if z_end is not None else self.centre[2] + self.height
        self.scan_points_per_z = max(2, int(scan_points_per_z))
        self.z_scans = max(1, int(z_scans))
        self.theta_start_deg = float(theta_start_deg)

    def _range(self, start, stop, step):
        """Return inclusive floating-point range values from start to stop."""
        values = []
        value = float(start)
        while value <= stop + 1e-9:
            values.append(value)
            value += step
        return values

    def _z_positions(self):
        """Compute Z layers either from calibration bounds or single-layer fallback."""
        if self.z_start == self.z_end:
            return [self.z_start]
        if self.z_scans <= 1:
            return [self.z_start]
        if self.z_step > 0:
            return self._range(self.z_start, self.z_end, (self.z_end - self.z_start) / max(1, self.z_scans - 1))
        return [self.z_start]

    def _wrap_yaw(self, yaw):
        """Wrap angle into [-180, 180] for initial yaw normalization."""
        return ((yaw + 180.0) % 360.0) - 180.0

    def _closest_equivalent_angle(self, target_deg, reference_deg):
        """Choose angle equivalent to target that is closest to previous command."""
        # Keep commanded yaw continuous by selecting the equivalent angle closest to the previous command.
        candidates = [target_deg - 720.0, target_deg - 360.0, target_deg, target_deg + 360.0, target_deg + 720.0]
        return min(candidates, key=lambda c: (abs(c - reference_deg), abs(c)))

    def _yaw_from_theta(self, theta_deg, previous_yaw):
        """Convert cylinder angle to tool yaw while preserving continuity between points."""
        base_yaw = theta_deg + 90.0
        if previous_yaw is None:
            return self._wrap_yaw(base_yaw)
        return self._closest_equivalent_angle(base_yaw, previous_yaw)

    def _angles_for_sweep(self, start_deg, end_deg, count):
        """Generate evenly spaced angular samples between two endpoints."""
        if count <= 1:
            return [start_deg]
        return [start_deg + (end_deg - start_deg) * i / (count - 1) for i in range(count)]

    def _min_arc_points_for_safe_chord(self, arc_deg=180.0):
        """Minimum arc waypoints so every chord between adjacent points stays outside the cylinder.

        The chord connecting two points on a circle of radius r_scan separated by angle Δθ has
        minimum distance r_scan·cos(Δθ/2) from the centre. We require that distance ≥ r_cyl,
        i.e. Δθ ≤ 2·arccos(r_cyl / r_scan).
        """
        r_scan = self.radius + self.lift_off
        r_cyl = self.radius
        if r_cyl <= 0 or r_scan <= r_cyl:
            # Fallback: dense enough for any reasonable geometry.
            return max(4, math.ceil(arc_deg / 30.0) + 1)
        max_step_deg = 2.0 * math.degrees(math.acos(r_cyl / r_scan))
        return max(2, math.ceil(arc_deg / max_step_deg) + 1)

    def generate(self):
        """Build full waypoint list with capture flags for scan and reset segments."""
        points = []
        r = self.radius + self.lift_off
        z_positions = self._z_positions()
        theta_offset_deg = self.theta_start_deg

        def append_segment_with_safety(segment_start_deg, segment_end_deg, capture_at_endpoints=True):
            """Append a segment with optional non-capture interpolation for safe chords."""
            safe_count = self._min_arc_points_for_safe_chord(abs(segment_end_deg - segment_start_deg))
            if safe_count <= 2:
                angles = [segment_start_deg, segment_end_deg]
            else:
                angles = self._angles_for_sweep(segment_start_deg, segment_end_deg, safe_count)

            for idx, theta_deg in enumerate(angles):
                capture = False
                if capture_at_endpoints and (idx == 0 or idx == len(angles) - 1):
                    capture = True
                segments.append((theta_deg, capture))

        def force_capture_at_indices(indices):
            """Mark selected segment indices as capture points."""
            for idx in indices:
                if 0 <= idx < len(segments):
                    theta_deg, _ = segments[idx]
                    segments[idx] = (theta_deg, True)

        # Radial standoff approach: move to the first scan angle but 50 mm further out,
        # so the robot approaches from clearly outside the cylinder on all axes.
        if z_positions:
            approach_r = r + 50.0
            approach_theta = math.radians(theta_offset_deg)
            ax = self.centre[0] + approach_r * math.cos(approach_theta)
            ay = self.centre[1] + approach_r * math.sin(approach_theta)
            az = z_positions[0]
            approach_yaw = self._wrap_yaw(theta_offset_deg + 90.0)
            points.append((ax, ay, az, 180.0, 0.0, approach_yaw, False))

        for z in z_positions:
            # Reset yaw continuity per Z layer so each ring starts from a neutral reference.
            previous_yaw = None
            segments = []

            # Build scan arcs with safety interpolation (all non-capture initially).
            first_start = len(segments)
            append_segment_with_safety(0.0, 180.0, capture_at_endpoints=False)
            first_end = len(segments)

            # Return to start side without recording.
            append_segment_with_safety(180.0, 0.0, capture_at_endpoints=False)
            if segments:
                # Avoid duplicate 180 endpoint at arc boundary.
                segments.pop(first_end)

            second_start = len(segments)
            append_segment_with_safety(0.0, -180.0, capture_at_endpoints=False)
            second_end = len(segments)

            # Return to start side without recording.
            append_segment_with_safety(-180.0, 0.0, capture_at_endpoints=False)
            if len(segments) > second_end:
                # Avoid duplicate -180 endpoint at arc boundary.
                segments.pop(second_end)

            # Enforce exact requested capture count per Z layer by selecting evenly spaced points
            # across the two scan arcs only.
            first_scan_indices = list(range(first_start, first_end))
            second_scan_indices = list(range(second_start, second_end))
            scan_indices = first_scan_indices + second_scan_indices
            if not scan_indices:
                scan_indices = list(range(first_start, first_end))

            requested = max(1, int(self.scan_points_per_z))
            if requested >= len(scan_indices):
                force_capture_at_indices(scan_indices)
            else:
                chosen = []
                for i in range(requested):
                    position = i * (len(scan_indices) - 1) / max(1, requested - 1)
                    idx = scan_indices[int(round(position))]
                    if not chosen or idx != chosen[-1]:
                        chosen.append(idx)
                while len(chosen) < requested:
                    for candidate in scan_indices:
                        if candidate not in chosen:
                            chosen.append(candidate)
                        if len(chosen) >= requested:
                            break
                force_capture_at_indices(chosen)

            for theta_deg, capture in segments:
                shifted_theta_deg = theta_deg + theta_offset_deg
                theta = math.radians(shifted_theta_deg)
                x = self.centre[0] + r * math.cos(theta)
                y = self.centre[1] + r * math.sin(theta)
                z_pos = z

                roll = 180.0
                pitch = 0.0
                yaw = self._yaw_from_theta(shifted_theta_deg, previous_yaw)
                previous_yaw = yaw

                # Store pose and capture flag used by runtime execution loop.
                points.append((x, y, z_pos, roll, pitch, yaw, capture))

        return points

    def save(self, filename):
        """Save planner configuration to disk as JSON."""
        payload = {
            "centre": self.centre,
            "radius": self.radius,
            "height": self.height,
            "lift_off": self.lift_off,
            "theta_step_deg": self.theta_step_deg,
            "z_step": self.z_step,
            "z_start": self.z_start,
            "z_end": self.z_end,
            "scan_points_per_z": self.scan_points_per_z,
            "z_scans": self.z_scans,
            "theta_start_deg": self.theta_start_deg,
        }
        Path(filename).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, filename):
        """Load planner configuration from JSON file."""
        payload = json.loads(Path(filename).read_text())
        return cls(
            centre=payload.get("centre", [0.0, 0.0, 0.0]),
            radius=payload.get("radius", 0.0),
            height=payload.get("height", 0.0),
            lift_off=payload.get("lift_off", 0.0),
            theta_step_deg=payload.get("theta_step_deg", 30.0),
            z_step=payload.get("z_step", 20.0),
            z_start=payload.get("z_start"),
            z_end=payload.get("z_end"),
            scan_points_per_z=payload.get("scan_points_per_z", 12),
            z_scans=payload.get("z_scans", 5),
            theta_start_deg=payload.get("theta_start_deg", 180.0),
        )


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


def _compute_cylinder_from_points(p0, p1, p2):
    """Compute cylinder geometry from three taught points."""
    x0, y0, z0, *_ = p0
    x1, y1, z1, *_ = p1
    x2, y2, z2, *_ = p2

    axis_x = x1 - x0
    axis_y = y1 - y0
    axis_z = z1 - z0
    axis_len = (axis_x ** 2 + axis_y ** 2 + axis_z ** 2) ** 0.5
    if axis_len == 0:
        raise ValueError("Invalid calibration: P1 must not equal P0")

    ux = axis_x / axis_len
    uy = axis_y / axis_len
    uz = axis_z / axis_len

    dx = x2 - x0
    dy = y2 - y0
    dz = z2 - z0
    t = dx * ux + dy * uy + dz * uz

    cx = x0 + ux * t
    cy = y0 + uy * t
    cz = z0 + uz * t

    rx = x2 - cx
    ry = y2 - cy
    rz = z2 - cz
    radius = (rx ** 2 + ry ** 2 + rz ** 2) ** 0.5
    if radius == 0:
        raise ValueError("Invalid calibration: P2 must not lie on cylinder axis")

    return [cx, cy, cz], radius


def _read_calibration_geometry(calibration_file):
    """Read full scan geometry from saved calibration touch points."""
    resolved_file = _resolve_calibration_file(calibration_file)
    if not resolved_file:
        return None

    try:
        payload = json.loads(Path(resolved_file).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    # ---- New format (4-point circle fit, saved directly) ----
    if "centre" in payload and "radius" in payload and "z_start" in payload:
        centre = [float(v) for v in payload["centre"]]
        radius = float(payload["radius"])
        z_start = float(payload["z_start"])
        z_end = float(payload["z_end"])
        # Use angle to the first raw touch point as scan start direction.
        raw_points = payload.get("raw_points", [])
        if raw_points:
            p0 = raw_points[0]
            start_theta_deg = math.degrees(math.atan2(float(p0[1]) - centre[1], float(p0[0]) - centre[0]))
        else:
            start_theta_deg = 0.0
        return {
            "resolved_file": resolved_file,
            "centre": centre,
            "radius": radius,
            "height": abs(z_end - z_start),
            "z_start": z_start,
            "z_end": z_end,
            "theta_start_deg": start_theta_deg,
        }

    # ---- Legacy format (raw p0/p1/p2 touch points) ----
    p0 = payload.get("p0")
    p1 = payload.get("p1")
    p2 = payload.get("p2")
    if p0 is None or p1 is None or p2 is None:
        return None

    centre, radius = _compute_cylinder_from_points(p0, p1, p2)
    start_theta_deg = math.degrees(math.atan2(float(p0[1]) - centre[1], float(p0[0]) - centre[0]))
    z0 = float(p0[2])
    z1 = float(p1[2])
    return {
        "resolved_file": resolved_file,
        "centre": centre,
        "radius": float(radius),
        "height": abs(z1 - z0),
        "z_start": min(z0, z1),
        "z_end": max(z0, z1),
        "theta_start_deg": float(start_theta_deg),
    }


def run_cylindrical_scan(
    centre,
    radius,
    height,
    lift_off,
    theta_step_deg=30.0,
    z_step=20.0,
    dwell_seconds=5.0,
    speed=40,
    output_folder="data/raw",
    calibration_file=None,
    scan_points_per_z=12,
    z_scans=5,
):
    """Execute cylindrical scan path and log synchronized EMAT + robot data."""
    calibration = _read_calibration_geometry(calibration_file)
    if calibration is not None:
        centre = calibration["centre"]
        radius = calibration["radius"]
        height = calibration["height"]
        z_start = calibration["z_start"]
        z_end = calibration["z_end"]
        theta_start_deg = calibration["theta_start_deg"]

        print(f"Using full geometry from {calibration['resolved_file']}")
        print(f"Calibrated centre: x={centre[0]:.1f}, y={centre[1]:.1f}, z={centre[2]:.1f}")
        print(f"Calibrated radius: {radius:.1f} mm")
        print(f"Calibrated z range: {z_start:.1f} mm to {z_end:.1f} mm")
        print(f"Calibrated start angle: {theta_start_deg:.1f} deg (from P0)")
    else:
        z_start = centre[2]
        z_end = centre[2] + height
        theta_start_deg = 180.0
        print("No valid calibration file found; using CLI/default geometry")

    planner = CylindricalScanPlanner(
        centre=centre,
        radius=radius,
        height=height,
        lift_off=lift_off,
        theta_step_deg=theta_step_deg,
        z_step=z_step,
        z_start=z_start,
        z_end=z_end,
        scan_points_per_z=scan_points_per_z,
        z_scans=z_scans,
        theta_start_deg=theta_start_deg,
    )

    points = planner.generate()
    capture_count = sum(1 for *_, capture in points if capture)
    reset_count = len(points) - capture_count

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
            print(f"Starting cylindrical scan with {len(points)} motion points")
            print(f"Capture points: {capture_count}, reset/approach points: {reset_count}")
            print(f"Z range: {z_start:.1f} mm to {z_end:.1f} mm")
            print(f"Using {scan_points_per_z} scan points per z layer and {z_scans} z layers")

            for index, (x, y, z, roll, pitch, yaw, capture) in enumerate(points, start=1):
                action = "SCAN" if capture else "RESET"
                print(f"[{index}/{len(points)}] {action} move to x={x:.1f}, y={y:.1f}, z={z:.1f}, yaw={yaw:.1f}")
                move_code = robot.move_to(x, y, z, speed=speed, roll=roll, pitch=pitch, yaw=yaw)
                if move_code == 9:
                    # Recover from transient "state not ready to move" by re-arming state once.
                    arm.set_state(0)
                    time.sleep(0.1)
                    move_code = robot.move_to(x, y, z, speed=speed, roll=roll, pitch=pitch, yaw=yaw)

                if move_code != 0:
                    state_code, state = arm.get_state()
                    err_code, err_warn = arm.get_err_warn_code(show=True)
                    raise RuntimeError(
                        "Move failed at point "
                        f"{index}/{len(points)} with API code={move_code}; "
                        f"state_query_code={state_code}, state={state}, "
                        f"err_query_code={err_code}, err_warn={err_warn}"
                    )

                if capture:
                    # Keep acquiring during dwell for a stable sample at each scan point.
                    dwell_until = time.monotonic() + dwell_seconds
                    data = None
                    while time.monotonic() < dwell_until:
                        data = emat.acquire()
                        plotter.update(data)
                        time.sleep(0.1)

                    pose = robot.get_pose()
                    logger.log(pose, data)
                    print(f"Captured point {index}/{len(points)}")

            print("Cylindrical scan complete")
    finally:
        plotter.close()
        logger.close()
        conn.disconnect()


if __name__ == "__main__":
    """CLI entry point for standalone cylindrical scan runs."""
    parser = argparse.ArgumentParser(description="Run a cylindrical EMAT scan around a vertical cylinder")
    parser.add_argument("--centre", type=float, nargs=3, default=[250.0, 0.0, 150.0], help="Cylinder centre x y z")
    parser.add_argument("--radius", type=float, default=50.0, help="Cylinder radius in mm")
    parser.add_argument("--height", type=float, default=150.0, help="Cylinder scan height in mm")
    #Modify lift off below.
    parser.add_argument("--lift-off", type=float, default=1.0, help="Radial lift-off from cylinder surface in mm")
    parser.add_argument("--dwell", type=float, default=5.0, help="Seconds to dwell at each point")
    parser.add_argument("--speed", type=float, default=40.0, help="Motion speed")
    parser.add_argument("--calibration-file", type=str, default=None, help="Optional saved calibration JSON file")
    parser.add_argument("--output-folder", type=str, default="data/raw", help="Folder for scan logs")
    parser.add_argument("--scan-points-per-z", type=int, default=None, help="Number of scan points per z layer")
    parser.add_argument("--z-scans", type=int, default=None, help="Number of z layers to scan")
    args = parser.parse_args()

    scan_points_per_z = args.scan_points_per_z
    if scan_points_per_z is None:
        scan_points_per_z = int(input("Number of scan points per z layer: ") or 12)

    z_scans = args.z_scans
    if z_scans is None:
        z_scans = int(input("Number of z layers to scan: ") or 5)

    run_cylindrical_scan(
        centre=args.centre,
        radius=args.radius,
        height=args.height,
        lift_off=args.lift_off,
        dwell_seconds=args.dwell,
        speed=args.speed,
        output_folder=args.output_folder,
        calibration_file=args.calibration_file,
        scan_points_per_z=scan_points_per_z,
        z_scans=z_scans,
    )
