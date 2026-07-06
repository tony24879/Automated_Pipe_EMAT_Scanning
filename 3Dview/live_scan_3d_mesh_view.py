"""PyVista-based live 3D mesh view for horizontal cylindrical scans."""

import math
import time
from pathlib import Path

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

        # Approximate geometry fallback when STLs are not available.
        self.base_height = 243.3
        self.link_upper = 200.0
        self.link_forearm = 190.0
        self.link_wrist_1 = 60.0
        self.link_wrist_2 = 55.0

        self.plotter = pv.Plotter(window_size=(1200, 820), title="Live Horizontal Scan 3D Mesh View")
        self.plotter.set_background("#f4f4f6")
        self.plotter.add_axes()
        self.plotter.show_grid(color="#bbbbbb")

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
        self.plotter.add_mesh(cylinder, color="#9b9b9b", opacity=0.2, smooth_shading=True)

    def _add_surface_points(self):
        if self.surface_points.size == 0:
            return
        surf_cloud = self.pv.PolyData(self.surface_points)
        self.plotter.add_mesh(
            surf_cloud,
            color="#2ca02c",
            point_size=6,
            render_points_as_spheres=True,
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
        self.plotter.add_mesh(self._line_mesh, color="#1f77b4", line_width=5)

        joints_cloud = self.pv.PolyData(zero_points)
        self.plotter.add_mesh(
            joints_cloud,
            color="#1f77b4",
            point_size=14,
            render_points_as_spheres=True,
        )
        self._joints_cloud = joints_cloud

        self._tcp_mesh = self.pv.PolyData(np.array([[0.0, 0.0, 0.0]], dtype=float))
        self.plotter.add_mesh(
            self._tcp_mesh,
            color="#d62728",
            point_size=15,
            render_points_as_spheres=True,
        )

        self._target_mesh = self.pv.PolyData(np.array([[0.0, 0.0, 0.0]], dtype=float))
        self.plotter.add_mesh(
            self._target_mesh,
            color="#ff7f0e",
            point_size=13,
            render_points_as_spheres=True,
        )

        self._trace_mesh = self._make_polyline(np.array([[0.0, 0.0, 0.0]], dtype=float))
        self.plotter.add_mesh(self._trace_mesh, color="#b0413e", line_width=2)

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
            actor = self.plotter.add_mesh(mesh, color="#d8dde3", opacity=0.95, smooth_shading=True)
            self._link_meshes[name] = mesh
            self._link_actors[name] = actor

    def _forward_kinematics_points(self, joint_angles_deg):
        angles = list(joint_angles_deg)[:6]
        if len(angles) < 6:
            angles.extend([0.0] * (6 - len(angles)))
        a = [math.radians(float(v)) for v in angles]

        origin = np.array([0.0, 0.0, 0.0], dtype=float)
        pos = origin.copy()
        rot = self._rot_z(a[0])

        points = [pos.copy()]

        pos = pos + rot @ np.array([0.0, 0.0, self.base_height], dtype=float)
        points.append(pos.copy())

        rot = rot @ self._rot_y(a[1])
        pos = pos + rot @ np.array([self.link_upper, 0.0, 0.0], dtype=float)
        points.append(pos.copy())

        rot = rot @ self._rot_y(a[2])
        pos = pos + rot @ np.array([self.link_forearm, 0.0, 0.0], dtype=float)
        points.append(pos.copy())

        rot = rot @ self._rot_x(a[3])
        pos = pos + rot @ np.array([self.link_wrist_1, 0.0, 0.0], dtype=float)
        points.append(pos.copy())

        rot = rot @ self._rot_y(a[4])
        pos = pos + rot @ np.array([self.link_wrist_2, 0.0, 0.0], dtype=float)
        points.append(pos.copy())

        return np.asarray(points, dtype=float)

    def _pose_from_arm(self, arm):
        pose_code, pose = arm.get_position()
        if pose_code != 0 or pose is None or len(pose) < 6:
            return None, None
        angles_code, angles = arm.get_servo_angle(is_radian=False)
        if angles_code != 0 or angles is None:
            angles = [0.0] * 6
        return pose, angles

    def update(self, tcp_pose, joint_angles_deg, current_target=None, capture=False):
        now = time.monotonic()
        if now - self._last_draw < self.draw_interval_sec:
            return
        self._last_draw = now

        points = self._forward_kinematics_points(joint_angles_deg)

        tcp_xyz = np.array([float(tcp_pose[0]), float(tcp_pose[1]), float(tcp_pose[2])], dtype=float)
        if np.linalg.norm(self.tcp_offset_xyz) > 1e-9:
            tcp_rot = self._rpy_deg_to_rot(tcp_pose[3], tcp_pose[4], tcp_pose[5])
            expected_joint6_xyz = tcp_xyz - tcp_rot @ self.tcp_offset_xyz
            shift_xyz = expected_joint6_xyz - points[-1]
            points = points + shift_xyz

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
        self._trace_mesh.overwrite(self._make_polyline(trace_points))

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
