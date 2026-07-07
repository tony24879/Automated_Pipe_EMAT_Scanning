"""PyVista-based live 3D mesh view for horizontal cylindrical scans."""

import math
import time
from pathlib import Path
import logging

import numpy as np


class LiveScan3DMeshView:
    """Realtime 3D mesh renderer with optional per-link STL loading."""

    def __init__(
        self,
        centre,
        radius,
        x_start,
        x_end,
        theta_limit_a_deg,
        theta_limit_b_deg,
        surface_points=None,
        tcp_offset_xyz=None,
        draw_interval_sec=0.05,
        mesh_dir=None,
        mesh_scale=1.0,
    ):
        try:
            import pyvista as pv
        except ImportError as exc:
            raise RuntimeError(
                "PyVista is not installed. Install with: pip install pyvista"
            ) from exc

        self.pv = pv
        self.centre = np.array(centre, dtype=float)
        self.radius = float(radius)
        self.x_start = float(x_start)
        self.x_end = float(x_end)
        self.theta_a = float(theta_limit_a_deg)
        self.theta_b = float(theta_limit_b_deg)
        self.surface_points = np.asarray(surface_points if surface_points is not None else [], dtype=float)
        self.tcp_offset_xyz = np.asarray(tcp_offset_xyz if tcp_offset_xyz is not None else [0.0, 0.0, 0.0], dtype=float)
        self.draw_interval_sec = max(0.01, float(draw_interval_sec))
        self.mesh_dir = Path(mesh_dir) if mesh_dir else None
        self.mesh_scale = float(mesh_scale)

        self._last_draw = 0.0
        self._trace_points = []
        self._line_mesh = None
        self._tcp_mesh = None
        self._target_mesh = None
        self._trace_mesh = None

        # User-provided zero-pose geometry (mm) and per-joint local offsets.
        self.base_origin = np.array([0.0, 0.0, 173.0], dtype=float)
        self.joint_offsets = [
            np.array([0.0, 78.0, 70.0], dtype=float),
            np.array([0.0, -33.0, 200.0], dtype=float),
            np.array([87.0, -44.7, -84.5], dtype=float),
            np.array([0.28382, 41.517, -142.60214], dtype=float),
            np.array([-0.00001, -41.01720, -41.00000], dtype=float),
        ]

        # Joint rotation axes in each joint's local frame.
        # Home-pose convention provided by user:
        # J1 +z, J2 +y, J3 -y, J4 -z, J5 +y, J6 -z.
        self.joint_axes = ["z", "y", "-y", "-z", "y", "-z"]

        # Keep model rooted in robot base frame unless explicitly changed.
        self.align_to_tcp = False

        # Lightweight tf-style composition for each downstream joint:
        # - "translate_then_rotate": T_offset @ R_joint
        # - "rotate_then_translate": R_joint @ T_offset
        self.joint_transform_order = "translate_then_rotate"

        self.plotter = pv.Plotter(window_size=(1200, 820), title="Live Horizontal Scan 3D Mesh View")
        self.plotter.set_background("#f4f4f6")
        self.plotter.add_axes()
        self.plotter.show_grid(color="#bbbbbb")

        # Reduce VTK noise and avoid more demanding shader paths on weaker drivers.
        logging.getLogger().setLevel(logging.ERROR)

        self._add_cylinder()
        self._add_surface_points()
        self._init_robot_visuals()
        self._set_camera()
        self.plotter.show(auto_close=False, interactive_update=True)

    def _rot_x(self, angle_rad):
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
            dtype=float,
        )

    def _rot_y(self, angle_rad):
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        return np.array(
            [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
            dtype=float,
        )

    def _rot_z(self, angle_rad):
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        return np.array(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )

    def _rot_axis(self, axis_name, angle_rad):
        axis = str(axis_name).strip().lower()
        sign = -1.0 if axis.startswith("-") else 1.0
        axis = axis.lstrip("+-")
        angle = sign * angle_rad
        if axis == "x":
            return self._rot_x(angle)
        if axis == "y":
            return self._rot_y(angle)
        return self._rot_z(angle)

    def _rpy_deg_to_rot(self, roll_deg, pitch_deg, yaw_deg):
        roll = math.radians(float(roll_deg))
        pitch = math.radians(float(pitch_deg))
        yaw = math.radians(float(yaw_deg))
        return self._rot_z(yaw) @ self._rot_y(pitch) @ self._rot_x(roll)

    def _set_camera(self):
        span_x = max(320.0, abs(self.x_end - self.x_start) + 260.0)
        span_yz = max(320.0, 2.0 * self.radius + 260.0)
        x_mid = 0.5 * (self.x_start + self.x_end)
        y_mid = self.centre[1]
        z_mid = self.centre[2]

        self.plotter.set_focus((x_mid, y_mid, z_mid))
        self.plotter.set_position((x_mid - span_x, y_mid - span_yz, z_mid + span_yz * 0.75))
        self.plotter.camera.up = (0.0, 0.0, 1.0)
        self.plotter.camera.clipping_range = (1.0, 6000.0)

    def _add_cylinder(self):
        x_mid = 0.5 * (self.x_start + self.x_end)
        height = max(1.0, abs(self.x_end - self.x_start))
        cylinder = self.pv.Cylinder(
            center=(x_mid, self.centre[1], self.centre[2]),
            direction=(1.0, 0.0, 0.0),
            radius=self.radius,
            height=height,
            resolution=80,
        )
        self.plotter.add_mesh(cylinder, color="#9b9b9b", opacity=0.2, smooth_shading=False, lighting=False)

    def _add_surface_points(self):
        if self.surface_points.size == 0:
            return
        surf_cloud = self.pv.PolyData(self.surface_points)
        self.plotter.add_mesh(
            surf_cloud,
            color="#2ca02c",
            point_size=6,
            render_points_as_spheres=False,
            lighting=False,
        )

    def _make_polyline(self, points):
        points = np.asarray(points, dtype=float)
        poly = self.pv.PolyData(points)
        if len(points) >= 2:
            line_cells = []
            for idx in range(len(points) - 1):
                line_cells.extend([2, idx, idx + 1])
            poly.lines = np.array(line_cells, dtype=np.int64)
        return poly

    def _init_robot_visuals(self):
        zero_points = np.zeros((6, 3), dtype=float)

        self._line_mesh = self._make_polyline(zero_points)
        self.plotter.add_mesh(self._line_mesh, color="#1f77b4", line_width=5, lighting=False)

        joints_cloud = self.pv.PolyData(zero_points)
        self.plotter.add_mesh(
            joints_cloud,
            color="#1f77b4",
            point_size=14,
            render_points_as_spheres=False,
            lighting=False,
        )
        self._joints_cloud = joints_cloud

        self._tcp_mesh = self.pv.PolyData(np.array([[0.0, 0.0, 0.0]], dtype=float))
        self.plotter.add_mesh(
            self._tcp_mesh,
            color="#d62728",
            point_size=15,
            render_points_as_spheres=False,
            lighting=False,
        )

        self._target_mesh = self.pv.PolyData(np.array([[0.0, 0.0, 0.0]], dtype=float))
        self.plotter.add_mesh(
            self._target_mesh,
            color="#ff7f0e",
            point_size=13,
            render_points_as_spheres=False,
            lighting=False,
        )

        self._trace_mesh = self._make_polyline(np.array([[0.0, 0.0, 0.0]], dtype=float))
        self.plotter.add_mesh(self._trace_mesh, color="#b0413e", line_width=2, lighting=False)

        self._init_mesh_actors()

    def _init_mesh_actors(self):
        self._link_meshes = {}
        self._link_actors = {}
        if not self.mesh_dir or not self.mesh_dir.exists():
            return

        names = ["base", "link1", "link2", "link3", "link4", "link5", "link6"]
        for name in names:
            mesh_path = self.mesh_dir / f"{name}.stl"
            if not mesh_path.exists():
                continue
            mesh = self.pv.read(str(mesh_path))
            if abs(self.mesh_scale - 1.0) > 1e-12:
                mesh = mesh.scale([self.mesh_scale, self.mesh_scale, self.mesh_scale], inplace=False)
            actor = self.plotter.add_mesh(mesh, color="#d8dde3", opacity=0.95, smooth_shading=False)
            self._link_meshes[name] = mesh
            self._link_actors[name] = actor

    def _homogeneous_transform(self, rotation, translation):
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        return transform

    def _translation_transform(self, translation):
        transform = np.eye(4, dtype=float)
        transform[:3, 3] = np.asarray(translation, dtype=float)
        return transform

    def _rotation_transform(self, axis_name, angle_rad):
        return self._homogeneous_transform(self._rot_axis(axis_name, angle_rad), np.zeros(3, dtype=float))

    def _compute_joint_positions_and_frames(self, joint_angles_deg):
        """Compute joint/link frames via explicit homogeneous transform chaining."""
        angles = list(joint_angles_deg)[:6]
        if len(angles) < 6:
            angles.extend([0.0] * (6 - len(angles)))
        a = [math.radians(float(v)) for v in angles]

        base_tf = self._translation_transform(self.base_origin)
        frames = {"base": base_tf}

        joint_points = []
        joint_tf = base_tf @ self._rotation_transform(self.joint_axes[0], a[0])
        frames["link1"] = joint_tf
        joint_points.append(joint_tf[:3, 3].copy())

        for idx in range(1, 6):
            offset_tf = self._translation_transform(self.joint_offsets[idx - 1])
            rot_tf = self._rotation_transform(self.joint_axes[idx], a[idx])

            if self.joint_transform_order == "rotate_then_translate":
                joint_tf = joint_tf @ rot_tf @ offset_tf
            else:
                joint_tf = joint_tf @ offset_tf @ rot_tf

            frames[f"link{idx + 1}"] = joint_tf
            joint_points.append(joint_tf[:3, 3].copy())

        return np.asarray(joint_points, dtype=float), frames

    def _align_frames_to_tcp(self, frames, tcp_pose):
        """Translate all frames so the approximated link6 frame matches live TCP minus tool offset."""
        if "link6" not in frames or np.linalg.norm(self.tcp_offset_xyz) <= 1e-9:
            return frames

        tcp_xyz = np.array([float(tcp_pose[0]), float(tcp_pose[1]), float(tcp_pose[2])], dtype=float)
        tcp_rot = self._rpy_deg_to_rot(tcp_pose[3], tcp_pose[4], tcp_pose[5])
        expected_link6_xyz = tcp_xyz - tcp_rot @ self.tcp_offset_xyz
        shift_xyz = expected_link6_xyz - frames["link6"][:3, 3]

        aligned = {}
        for name, transform in frames.items():
            shifted = np.array(transform, copy=True)
            shifted[:3, 3] += shift_xyz
            aligned[name] = shifted
        return aligned

    def _apply_link_actor_transforms(self, frames):
        for name, actor in self._link_actors.items():
            transform = frames.get(name)
            if transform is None:
                continue
            actor.user_matrix = transform

    def _pose_from_arm(self, arm):
        pose_code, pose = arm.get_position()
        if pose_code != 0 or pose is None or len(pose) < 6:
            return None, None

        angles = None
        try:
            state_code, state_payload = arm.get_joint_states(is_radian=False, num=1)
            if state_code == 0 and isinstance(state_payload, (list, tuple)) and len(state_payload) >= 1:
                candidate = state_payload[0]
                if isinstance(candidate, (list, tuple)) and len(candidate) >= 6:
                    angles = list(candidate[:6])
        except Exception:
            angles = None

        if angles is None:
            angles_code, angles = arm.get_servo_angle(is_radian=False)
            if angles_code != 0 or angles is None:
                angles = [0.0] * 6

        return pose, angles

    def update(self, tcp_pose, joint_angles_deg, current_target=None, capture=False):
        now = time.monotonic()
        if now - self._last_draw < self.draw_interval_sec:
            return
        self._last_draw = now

        points, frames = self._compute_joint_positions_and_frames(joint_angles_deg)

        tcp_xyz = np.array([float(tcp_pose[0]), float(tcp_pose[1]), float(tcp_pose[2])], dtype=float)
        if self.align_to_tcp and np.linalg.norm(self.tcp_offset_xyz) > 1e-9:
            tcp_rot = self._rpy_deg_to_rot(tcp_pose[3], tcp_pose[4], tcp_pose[5])
            expected_joint6_xyz = tcp_xyz - tcp_rot @ self.tcp_offset_xyz
            shift_xyz = expected_joint6_xyz - points[-1]
            points = points + shift_xyz
            frames = self._align_frames_to_tcp(frames, tcp_pose)

        self._apply_link_actor_transforms(frames)

        self._line_mesh.points = points
        self._joints_cloud.points = points

        self._tcp_mesh.points = np.array([tcp_xyz], dtype=float)

        if current_target is not None:
            target_xyz = np.array([[float(current_target[0]), float(current_target[1]), float(current_target[2])]], dtype=float)
            self._target_mesh.points = target_xyz

        self._trace_points.append(tcp_xyz)
        if len(self._trace_points) > 800:
            self._trace_points = self._trace_points[-800:]

        trace_points = np.asarray(self._trace_points, dtype=float)
        self._trace_mesh.copy_from(self._make_polyline(trace_points), deep=False)

        # STL link actors are loaded and shown as static placeholders until per-link
        # frame transforms are provided with robot-specific zero-pose calibration.
        self.plotter.render()
        self.plotter.update()

    def update_from_arm(self, arm, current_target=None, capture=False):
        pose, angles = self._pose_from_arm(arm)
        if pose is None:
            return
        self.update(pose, angles, current_target=current_target, capture=capture)

    def close(self):
        if self.plotter is not None:
            self.plotter.close()
