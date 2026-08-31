"""Plan and execute EMAT scans around a horizontal cylinder with serpentine theta sweeps."""

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

from config.robot_config import ROBOT_IP, TCP_OFFSET
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
        emat_captures_per_point=1,
        num_repeats=1,
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
        self.emat_captures_per_point = max(1, int(emat_captures_per_point))
        self.num_repeats = max(1, int(num_repeats))

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
        """Build serpentine theta sweeps with repeats and inter-scan transitions."""
        all_points = []
        r_scan = self.radius + self.lift_off
        r_outer = r_scan + self.outer_offset_mm

        x_positions = self._x_positions()
        if not x_positions:
            return all_points

        theta_a = self.theta_limit_a_deg
        theta_b = self._unwrap_near(self.theta_limit_b_deg, theta_a)

        # Generate a single scan
        def generate_single_scan():
            """Generate one complete scan."""
            points = []
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
            previous_roll = self._append_point(points, r_scan, theta_a, first_x, previous_roll, capture=self.emat_captures_per_point)

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
                    previous_roll = self._append_point(points, r_scan, sweep_start, x_mm, previous_roll, capture=self.emat_captures_per_point)

                for i in range(len(sweep_thetas) - 1):
                    previous_roll = move_inner_to_inner(
                        sweep_thetas[i],
                        sweep_thetas[i + 1],
                        x_mm,
                        previous_roll,
                        capture_to=self.emat_captures_per_point,
                    )

                prev_x = x_mm

            return points

        # Generate the first scan
        all_points = generate_single_scan()
        last_roll_state = all_points[-1][3] if all_points else None  # Extract roll from last point
        
        # For repeats, add transition moves and subsequent scans
        for repeat_index in range(1, self.num_repeats):
            if not all_points:
                break
            
            # Get the last point of current scan (the actual contact point)
            last_scan_point = all_points[-1]
            last_x, last_y, last_z, last_roll, last_pitch, last_yaw = last_scan_point[:6]
            
            # Get the first contact point of the next scan (should be at first x position, theta_a)
            first_x = x_positions[0]
            first_theta = theta_a
            first_theta_rad = math.radians(first_theta)
            first_y = self.centre[1] + r_scan * math.cos(first_theta_rad)
            first_z = self.centre[2] + r_scan * math.sin(first_theta_rad)

            # Outer-ring position above the first scan point — descend here, then move radially in.
            first_outer_y = self.centre[1] + r_outer * math.cos(first_theta_rad)
            first_outer_z = self.centre[2] + r_outer * math.sin(first_theta_rad)

            # 1. Move radially outward from the final point of current scan
            radial_retreat_mm = 25.0
            dy = last_y - self.centre[1]
            dz = last_z - self.centre[2]
            radial_norm = math.hypot(dy, dz)
            if radial_norm < 1e-9:
                radial_out_y = last_y + radial_retreat_mm
                radial_out_z = last_z
            else:
                radial_out_y = last_y + radial_retreat_mm * (dy / radial_norm)
                radial_out_z = last_z + radial_retreat_mm * (dz / radial_norm)

            # 2. Lift to safe Z (clearance above the work)
            safe_z = max(last_z, first_outer_z) + 80.0

            # 3. Get the roll orientation for first point of next scan
            first_roll = self._roll_from_theta(first_theta, last_roll)

            # Build the transition sequence — transit XY to above the outer ring (not the
            # scan surface) so the Z descent stays clear of the cylinder, then move radially in.
            transition_moves = [
                (last_x, radial_out_y, radial_out_z, last_roll, last_pitch, last_yaw, False),  # Radial outward
                (last_x, radial_out_y, safe_z, last_roll, last_pitch, last_yaw, False),  # Lift to clearance
                (last_x, radial_out_y, safe_z, first_roll, 0.0, 0.0, False),  # Realign tool orientation
                (first_x, first_outer_y, safe_z, first_roll, 0.0, 0.0, False),  # Transit XY to above outer ring
                (first_x, first_outer_y, first_outer_z, first_roll, 0.0, 0.0, False),  # Descend to outer ring height
            ]

            # Add transition moves to all points
            all_points.extend(transition_moves)

            # Generate next scan. Skip [0]=standoff and [1]=outer_ring since the transition
            # already leaves the arm at the outer ring; start at [2] (first scan capture).
            next_scan = generate_single_scan()
            if next_scan:
                all_points.extend(next_scan[2:])
            
            # Update last_roll_state for next iteration
            if all_points:
                last_roll_state = all_points[-1][3]

        return all_points

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
            "emat_captures_per_point": self.emat_captures_per_point,
            "num_repeats": self.num_repeats,
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

    acceptance_gate = payload.get("acceptance_gate") if isinstance(payload, dict) else None
    gate_passed = True
    gate_failures = []
    if isinstance(acceptance_gate, dict):
        gate_passed = bool(acceptance_gate.get("passed", True))
        failures = acceptance_gate.get("failures", [])
        if isinstance(failures, list):
            gate_failures = [str(item) for item in failures]

    fit_version = str(payload.get("fit_version", "unknown"))

    return {
        "resolved_file": resolved_file,
        "centre": centre,
        "radius": radius,
        "length": abs(x_end - x_start),
        "x_start": x_start,
        "x_end": x_end,
        "theta_limit_a_deg": theta_a,
        "theta_limit_b_deg": theta_b,
        "fit_version": fit_version,
        "acceptance_gate_passed": gate_passed,
        "acceptance_gate_failures": gate_failures,
    }


def _load_live_scan_3d_view_class():
    """Load the optional live 3D view wrapper from the 3Dview folder."""
    module_path = Path(__file__).resolve().parent.parent / "3Dview" / "live_scan_3d_view.py"
    if not module_path.exists():
        return None

    spec = importlib.util.spec_from_file_location("live_scan_3d_view", module_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "LiveScan3DView", None)


def _load_live_scan_3d_mesh_view_class():
    """Load the optional PyVista mesh-based 3D view wrapper from the 3Dview folder."""
    module_path = Path(__file__).resolve().parent.parent / "3Dview" / "live_scan_3d_mesh_view.py"
    if not module_path.exists():
        return None

    spec = importlib.util.spec_from_file_location("live_scan_3d_mesh_view", module_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "LiveScan3DMeshView", None)


def _create_3d_view(
    backend,
    centre,
    radius,
    x_start,
    x_end,
    theta_limit_a_deg,
    theta_limit_b_deg,
    cylinder_surface_points,
    mesh_dir,
    mesh_scale,
):
    """Create a live 3D view using selected backend with fallback behavior."""
    base_kwargs = {
        "centre": centre,
        "radius": radius,
        "x_start": x_start,
        "x_end": x_end,
        "theta_limit_a_deg": theta_limit_a_deg,
        "theta_limit_b_deg": theta_limit_b_deg,
        "surface_points": cylinder_surface_points,
        "tcp_offset_xyz": TCP_OFFSET[:3],
    }

    if backend in ("mesh", "auto"):
        mesh_view_class = _load_live_scan_3d_mesh_view_class()
        if mesh_view_class is not None:
            try:
                view = mesh_view_class(
                    **base_kwargs,
                    mesh_dir=mesh_dir,
                    mesh_scale=mesh_scale,
                )
                print("Live 3D mesh view enabled")
                return view
            except Exception as exc:
                if backend == "mesh":
                    raise RuntimeError(f"Unable to initialize mesh backend: {exc}") from exc
                print(f"Warning: mesh backend unavailable ({exc}); falling back to matplotlib backend")
        elif backend == "mesh":
            raise RuntimeError("Mesh backend requested but 3D mesh wrapper file is missing")

    if backend in ("matplot", "auto"):
        matplot_view_class = _load_live_scan_3d_view_class()
        if matplot_view_class is not None:
            try:
                view = matplot_view_class(**base_kwargs)
                print("Live 3D matplotlib view enabled")
                return view
            except Exception as exc:
                raise RuntimeError(f"Unable to initialize matplotlib backend: {exc}") from exc

    raise RuntimeError("No 3D backend could be loaded")


def _build_cylinder_surface_points_from_scan(points, centre, lift_off):
    """Project capture points inward by lift-off to estimate cylinder surface points."""
    centre_y = float(centre[1])
    centre_z = float(centre[2])
    projected = []

    for x, y, z, *_rest, capture in points:
        if not capture:
            continue

        dy = float(y) - centre_y
        dz = float(z) - centre_z
        radial_norm = math.hypot(dy, dz)
        if radial_norm < 1e-9:
            continue

        uy = dy / radial_norm
        uz = dz / radial_norm

        projected.append(
            (
                float(x),
                float(y) - float(lift_off) * uy,
                float(z) - float(lift_off) * uz,
            )
        )

    return projected


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
    emat_captures_per_point=1,
    num_repeats=1,
    theta_limit_a_deg=0.0,
    theta_limit_b_deg=180.0,
    enable_3d_view=True,
    view_3d_backend="auto",
    robot_mesh_dir="3Dview/meshes/lite6",
    robot_mesh_scale=1.0,
    allow_failed_calibration=False,
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
        gate_passed = bool(calibration.get("acceptance_gate_passed", True))
        gate_failures = calibration.get("acceptance_gate_failures", []) or []
        fit_version = calibration.get("fit_version", "unknown")

        print(f"Using horizontal geometry from {calibration['resolved_file']}")
        print(f"Calibration fit version: {fit_version}")
        print(f"Calibrated centre: x={centre[0]:.1f}, y={centre[1]:.1f}, z={centre[2]:.1f}")
        print(f"Calibrated radius: {radius:.1f} mm")
        print(f"Calibrated x range: {x_start:.1f} mm to {x_end:.1f} mm")
        print(f"Theta limits: {theta_limit_a_deg:.1f} deg -> {theta_limit_b_deg:.1f} deg")

        # Respect calibration acceptance gating so failed geometry does not
        # silently drive hardware into unsafe or low-quality trajectories.
        if not gate_passed:
            print("Warning: calibration acceptance gate status is FAIL")
            for failure in gate_failures:
                print(f"  - {failure}")
            if not allow_failed_calibration:
                raise RuntimeError(
                    "Refusing to run scan with failed calibration. "
                    "Re-run calibration or pass --allow-failed-calibration to override."
                )
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
        emat_captures_per_point=emat_captures_per_point,
        num_repeats=num_repeats,
    )

    points = planner.generate()
    cylinder_surface_points = _build_cylinder_surface_points_from_scan(points, centre, lift_off)

    conn = RobotConnection(ROBOT_IP)
    arm = conn.connect()
    robot = Lite6(arm)
    setup = RobotSetup(arm)
    setup.configure()

    plotter = None
    try:
        plotter = LiveWaveformPlot()
    except Exception as exc:  # noqa: BLE001 - live plot is optional and must not block scan execution.
        print(f"Warning: unable to initialize live waveform plot ({exc}); continuing without live plot")
    view3d = None
    if enable_3d_view:
        try:
            view3d = _create_3d_view(
                backend=view_3d_backend,
                centre=centre,
                radius=radius,
                x_start=x_start,
                x_end=x_end,
                theta_limit_a_deg=theta_limit_a_deg,
                theta_limit_b_deg=theta_limit_b_deg,
                cylinder_surface_points=cylinder_surface_points,
                mesh_dir=robot_mesh_dir,
                mesh_scale=robot_mesh_scale,
            )
        except Exception as exc:  # noqa: BLE001 - 3D view is optional for scan operation.
            print(f"Warning: unable to initialize 3D scan view ({exc})")

    logger = SyncLogger(folder=output_folder)

    startup_origin_pose = None
    startup_safe_z = None
    last_reached_pose = None

    def _move_with_retry(x, y, z, roll, pitch, yaw, move_speed):
        """Execute one Cartesian move and retry once if controller state is not ready."""
        move_code = robot.move_to(x, y, z, speed=move_speed, roll=roll, pitch=pitch, yaw=yaw)
        if move_code == 9:
            arm.set_state(0)
            time.sleep(0.1)
            move_code = robot.move_to(x, y, z, speed=move_speed, roll=roll, pitch=pitch, yaw=yaw)
        return move_code

    def _execute_reverse_startup_retreat(reason):
        """Retreat with lift/across transit, then run controller reset."""
        if startup_origin_pose is None or startup_safe_z is None:
            print("Warning: startup retreat context unavailable; skipping reverse retreat")
            return

        retreat_pose = last_reached_pose
        if retreat_pose is None:
            current_pose = robot.get_pose()
            if current_pose and len(current_pose) >= 6:
                retreat_pose = tuple(float(v) for v in current_pose[:6])

        if retreat_pose is None:
            print("Warning: unable to determine current pose for retreat; skipping reverse retreat")
            return

        origin_x, origin_y, _origin_z, origin_roll, origin_pitch, origin_yaw = startup_origin_pose
        current_x, current_y, current_z, current_roll, current_pitch, current_yaw = retreat_pose
        safe_z = float(startup_safe_z)
        retreat_speed = min(float(speed), 30.0)
        radial_retreat_mm = 25.0

        dy = current_y - float(centre[1])
        dz = current_z - float(centre[2])
        radial_norm = math.hypot(dy, dz)
        if radial_norm < 1e-9:
            radial_out_y = current_y + radial_retreat_mm
            radial_out_z = current_z
        else:
            radial_out_y = current_y + radial_retreat_mm * (dy / radial_norm)
            radial_out_z = current_z + radial_retreat_mm * (dz / radial_norm)

        # Sequence is ordered to first create radial separation from the work
        # surface, then regain Z clearance, then traverse at safe height.
        retreat_moves = [
            (current_x, radial_out_y, radial_out_z, current_roll, current_pitch, current_yaw, "Radial outward nudge"),
            (current_x, radial_out_y, safe_z, current_roll, current_pitch, current_yaw, "Lift to clearance"),
            (current_x, radial_out_y, safe_z, origin_roll, origin_pitch, origin_yaw, "Realign tool to startup down-Z orientation"),
            (origin_x, origin_y, safe_z, origin_roll, origin_pitch, origin_yaw, "Transit XY at clearance"),
        ]

        print(
            f"Executing reverse startup retreat after {reason}: "
            f"radial-outward, up to z={safe_z:.1f}, realign to down-Z, across to startup XY, then controller reset"
        )

        for rx, ry, rz, rroll, rpitch, ryaw, label in retreat_moves:
            move_code = _move_with_retry(rx, ry, rz, rroll, rpitch, ryaw, retreat_speed)
            if move_code != 0:
                print(f"Warning: retreat step '{label}' failed with API code={move_code}; aborting retreat")
                return

        print("Retreat transit complete; running arm.reset(wait=True)")
        reset_code = arm.reset(wait=True)
        if reset_code is None:
            state_query_code, state = arm.get_state()
            err_query_code, err_warn = arm.get_err_warn_code(show=True)
            if state_query_code == 0 and err_query_code == 0 and err_warn == [0, 0]:
                print("Robot reset completed")
            else:
                print(
                    "Warning: arm.reset(wait=True) returned None and controller status is not clean; "
                    f"state_query_code={state_query_code}, state={state}, "
                    f"err_query_code={err_query_code}, err_warn={err_warn}"
                )
        elif reset_code != 0:
            print(f"Warning: arm.reset(wait=True) failed with API code={reset_code}")
        else:
            print("Robot reset completed")

    try:
        with EMATSession() as emat:
            print("Configuring EMAT...")
            emat.configure()
            execution_points = list(points)
            scan_completed = False
            scan_interrupted = False
            pending_exception = None

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
                        startup_origin_pose = (
                            current_x,
                            current_y,
                            current_z,
                            current_roll,
                            current_pitch,
                            current_yaw,
                        )
                        first_x, first_y, first_z, first_roll, first_pitch, first_yaw, _ = points[0]

                        startup_clearance_mm = 80.0
                        safe_z = max(current_z, first_z + startup_clearance_mm)
                        startup_safe_z = float(safe_z)

                        z_lift_waypoint = (current_x, current_y, safe_z, current_roll, current_pitch, current_yaw, False)
                        xy_transit_waypoint = (first_x, first_y, safe_z, current_roll, current_pitch, current_yaw, False)
                        rotate_at_clearance_waypoint = (first_x, first_y, safe_z, first_roll, first_pitch, first_yaw, False)

                        execution_points = [z_lift_waypoint, xy_transit_waypoint, rotate_at_clearance_waypoint] + points
                        print(
                            "Safer startup transit enabled: "
                            f"lift to z={safe_z:.1f} mm, XY at clearance, then rotate before descent"
                        )
                except Exception as exc:  # noqa: BLE001 - fallback to original first move on split-planning failure.
                    print(f"Warning: unable to create split first move ({exc}); using original path")

            capture_count = sum(1 for *_, capture in execution_points if capture)
            reset_count = len(execution_points) - capture_count
            first_capture_x = None

            print(f"Starting horizontal cylindrical scan with {len(execution_points)} motion points")
            print(f"Capture points: {capture_count}, reset/approach points: {reset_count}")
            print(f"X range: {x_start:.1f} mm to {x_end:.1f} mm")
            print(f"Using {scan_points_per_x} scan points per x layer, {x_scans} x layers, {emat_captures_per_point} capture(s) per point, and {num_repeats} repeat(s)")

            if view3d is not None:
                view3d.update_from_arm(arm)

            try:
                for index, (x, y, z, roll, pitch, yaw, capture) in enumerate(execution_points, start=1):
                    # capture can be False (0), True (1), or an integer > 1
                    is_capture = bool(capture)
                    num_captures = int(capture) if capture else 0
                    action = "SCAN" if is_capture else "RESET"
                    print(
                        f"[{index}/{len(execution_points)}] {action} move to "
                        f"x={x:.1f}, y={y:.1f}, z={z:.1f}, roll={roll:.1f}, pitch={pitch:.1f}, yaw={yaw:.1f}"
                    )
                    move_code = _move_with_retry(x, y, z, roll, pitch, yaw, speed)

                    if move_code != 0:
                        state_code, state = arm.get_state()
                        err_code, err_warn = arm.get_err_warn_code(show=True)
                        raise RuntimeError(
                            "Move failed at point "
                            f"{index}/{len(execution_points)} with API code={move_code}; "
                            f"state_query_code={state_code}, state={state}, "
                            f"err_query_code={err_code}, err_warn={err_warn}"
                        )

                    last_reached_pose = (float(x), float(y), float(z), float(roll), float(pitch), float(yaw))

                    if view3d is not None:
                        view3d.update_from_arm(arm, current_target=(x, y, z), capture=is_capture)

                    if is_capture:
                        # Perform multiple captures at this point if needed
                        for capture_num in range(num_captures):
                            dwell_until = time.monotonic() + dwell_seconds
                            data = None
                            while time.monotonic() < dwell_until:
                                data = emat.acquire()
                                if plotter is not None:
                                    plotter.update(data)
                                if view3d is not None:
                                    view3d.update_from_arm(arm, current_target=(x, y, z), capture=True)
                                time.sleep(0.1)

                            if first_capture_x is None:
                                first_capture_x = float(x)

                            # Theta/axis_position are scan coordinates exported for
                            # downstream ToF heatmaps and section-aligned analysis.
                            theta_deg = math.degrees(math.atan2(float(z) - float(centre[2]), float(y) - float(centre[1])))
                            axis_position_mm = float(x) - float(first_capture_x)

                            pose = robot.get_pose()
                            logger.log(pose, data, theta=theta_deg, axis_position=axis_position_mm)
                            print(f"Captured point {index}/{len(execution_points)} (capture {capture_num + 1}/{num_captures})")

                scan_completed = True
                print("Horizontal cylindrical scan complete")
            except KeyboardInterrupt:
                scan_interrupted = True
                print("Scan interrupted by user")
            except Exception as exc:  # noqa: BLE001 - keep failure for deferred handling while preserving retreat path.
                pending_exception = exc

            if scan_completed or scan_interrupted:
                retreat_reason = "completion" if scan_completed else "interruption"
                try:
                    _execute_reverse_startup_retreat(retreat_reason)
                except Exception as retreat_exc:  # noqa: BLE001 - retreat failure is non-fatal during shutdown handling.
                    print(f"Warning: reverse retreat failed ({retreat_exc})")

            if pending_exception is not None:
                raise pending_exception
    finally:
        if plotter is not None:
            plotter.close()
        if view3d is not None:
            view3d.close()
        logger.close()
        conn.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a cylindrical EMAT scan around a horizontal cylinder")
    parser.add_argument("--centre", type=float, nargs=3, default=[250.0, 0.0, 150.0], help="Cylinder centre x y z")
    parser.add_argument("--radius", type=float, default=50.0, help="Cylinder radius in mm")
    parser.add_argument("--length", type=float, default=150.0, help="Cylinder scan length along x in mm")
    parser.add_argument("--lift-off", type=float, default=0.0, help="Radial lift-off from cylinder surface in mm")
    parser.add_argument("--outer-offset-mm", type=float, default=20.0, help="Extra radial offset for transition ring in mm")
    parser.add_argument("--theta-limit-a-deg", type=float, default=0.0, help="First angular limit in yz-plane degrees")
    parser.add_argument("--theta-limit-b-deg", type=float, default=180.0, help="Second angular limit in yz-plane degrees")
    parser.add_argument("--dwell", type=float, default=0.5, help="Seconds to dwell at each point")
    parser.add_argument("--speed", type=float, default=40.0, help="Motion speed")
    parser.add_argument("--calibration-file", type=str, default=None, help="Optional saved calibration JSON file")
    parser.add_argument("--output-folder", type=str, default="data/raw", help="Folder for scan logs")
    parser.add_argument("--scan-points-per-x", type=int, default=None, help="Number of scan points per x layer")
    parser.add_argument("--x-scans", type=int, default=None, help="Number of x layers to scan")
    parser.add_argument("--emat-captures", type=int, default=1, help="Number of EMAT captures per scan point")
    parser.add_argument("--num-repeats", type=int, default=1, help="Number of times to repeat the entire scan")
    parser.add_argument("--disable-3d-view", action="store_true", help="Disable live 3D visualization window")
    parser.add_argument(
        "--view-3d-backend",
        type=str,
        default="auto",
        choices=["auto", "matplot", "mesh"],
        help="3D view backend selection",
    )
    parser.add_argument(
        "--robot-mesh-dir",
        type=str,
        default="3Dview/meshes/lite6",
        help="Folder containing per-link STL files for mesh backend",
    )
    parser.add_argument(
        "--robot-mesh-scale",
        type=float,
        default=1.0,
        help="Uniform STL scale factor for mesh backend",
    )
    parser.add_argument(
        "--allow-failed-calibration",
        action="store_true",
        help="Allow scan to continue even if calibration acceptance gate failed",
    )
    args = parser.parse_args()

    scan_points_per_x = args.scan_points_per_x
    if scan_points_per_x is None:
        scan_points_per_x = int(input("Number of scan points per x layer: ") or 12)

    x_scans = args.x_scans
    if x_scans is None:
        x_scans = int(input("Number of x layers to scan: ") or 5)

    emat_captures = args.emat_captures
    if emat_captures is None or emat_captures < 1:
        emat_captures = int(input("Number of EMAT captures per point: ") or 1)

    num_repeats = args.num_repeats
    if num_repeats is None or num_repeats < 1:
        num_repeats = int(input("Number of scan repeats: ") or 1)

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
        emat_captures_per_point=emat_captures,
        num_repeats=num_repeats,
        theta_limit_a_deg=args.theta_limit_a_deg,
        theta_limit_b_deg=args.theta_limit_b_deg,
        enable_3d_view=not args.disable_3d_view,
        view_3d_backend=args.view_3d_backend,
        robot_mesh_dir=args.robot_mesh_dir,
        robot_mesh_scale=args.robot_mesh_scale,
        allow_failed_calibration=args.allow_failed_calibration,
    )
