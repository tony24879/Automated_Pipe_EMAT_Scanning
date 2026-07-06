"""Live 3D visualization for horizontal cylindrical scans."""

import math
import time

import matplotlib.pyplot as plt
import numpy as np


def _rot_x(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=float,
    )


def _rot_y(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=float,
    )


def _rot_z(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def _rpy_deg_to_rot(roll_deg, pitch_deg, yaw_deg):
    """Convert roll-pitch-yaw (deg) to a tool-to-base rotation matrix."""
    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    yaw = math.radians(float(yaw_deg))
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)


class LiveScan3DView:
    """Realtime 3D view of robot pose and horizontal-cylinder scan surface points."""

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
        **_unused_kwargs,
    ):
        self.centre = np.array(centre, dtype=float)
        self.radius = float(radius)
        self.x_start = float(x_start)
        self.x_end = float(x_end)
        self.theta_a = float(theta_limit_a_deg)
        self.theta_b = float(theta_limit_b_deg)
        self.surface_points = np.asarray(surface_points if surface_points is not None else [], dtype=float)
        self.tcp_offset_xyz = np.asarray(tcp_offset_xyz if tcp_offset_xyz is not None else [0.0, 0.0, 0.0], dtype=float)
        self.draw_interval_sec = max(0.01, float(draw_interval_sec))

        self._trace_points = []
        self._last_draw = 0.0

        # Approximate Lite6 link lengths (mm) for a lightweight 6-joint stick-model visualization.
        self.base_height = 243.3
        self.link_upper = 200.0
        self.link_forearm = 190.0
        self.link_wrist_1 = 60.0
        self.link_wrist_2 = 55.0

        plt.ion()
        self.fig = plt.figure(figsize=(9, 7))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self._setup_axes()

        self.robot_line, = self.ax.plot([], [], [], "-o", lw=2.5, color="#1f77b4", markersize=5, label="6 joints")
        self.tcp_dot = self.ax.scatter([], [], [], s=55, c="#d62728", label="TCP")
        self.target_dot = self.ax.scatter([], [], [], s=45, c="#ff7f0e", marker="x", label="Current target")
        self.trace_line, = self.ax.plot([], [], [], "-", lw=1.5, alpha=0.75, color="#d62728", label="TCP trace")

        self._draw_cylinder_wireframe()
        self._draw_surface_points()
        self.ax.legend(loc="upper left")

    def _setup_axes(self):
        self.ax.set_title("Live Horizontal Scan 3D View")
        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")
        self.ax.set_zlabel("Z (mm)")
        self.ax.view_init(elev=28.0, azim=-52.0)

        span_x = max(320.0, abs(self.x_end - self.x_start) + 260.0)
        span_yz = max(320.0, 2.0 * self.radius + 260.0)

        x_mid = 0.5 * (self.x_start + self.x_end)
        y_mid = self.centre[1]
        z_mid = self.centre[2]

        self.ax.set_xlim(x_mid - 0.5 * span_x, x_mid + 0.5 * span_x)
        self.ax.set_ylim(y_mid - 0.5 * span_yz, y_mid + 0.5 * span_yz)
        self.ax.set_zlim(z_mid - 0.5 * span_yz, z_mid + 0.5 * span_yz)
        self.ax.set_box_aspect((1.2, 1.0, 1.0))

    def _draw_cylinder_wireframe(self):
        theta = np.deg2rad(np.linspace(self.theta_a, self.theta_b, 40))
        xs = np.linspace(self.x_start, self.x_end, 20)
        tt, xx = np.meshgrid(theta, xs)

        yy = self.centre[1] + self.radius * np.cos(tt)
        zz = self.centre[2] + self.radius * np.sin(tt)
        self.ax.plot_wireframe(xx, yy, zz, rstride=2, cstride=4, linewidth=0.6, color="#7f7f7f", alpha=0.35)

    def _draw_surface_points(self):
        if self.surface_points.size == 0:
            return
        self.ax.scatter(
            self.surface_points[:, 0],
            self.surface_points[:, 1],
            self.surface_points[:, 2],
            s=16,
            c="#2ca02c",
            alpha=0.85,
            label="Estimated cylinder points",
        )

    def _forward_kinematics_points(self, joint_angles_deg):
        angles = list(joint_angles_deg)[:6]
        if len(angles) < 6:
            angles.extend([0.0] * (6 - len(angles)))
        a = [math.radians(float(v)) for v in angles]

        origin = np.array([0.0, 0.0, 0.0], dtype=float)

        pos = origin.copy()
        rot = _rot_z(a[0])

        # Joint 1 (base yaw axis location)
        points = [pos.copy()]

        # Joint 2
        pos = pos + rot @ np.array([0.0, 0.0, self.base_height], dtype=float)
        points.append(pos.copy())

        rot = rot @ _rot_y(a[1])
        # Joint 3
        pos = pos + rot @ np.array([self.link_upper, 0.0, 0.0], dtype=float)
        points.append(pos.copy())

        rot = rot @ _rot_y(a[2])
        # Joint 4
        pos = pos + rot @ np.array([self.link_forearm, 0.0, 0.0], dtype=float)
        points.append(pos.copy())

        rot = rot @ _rot_x(a[3])
        # Joint 5
        pos = pos + rot @ np.array([self.link_wrist_1, 0.0, 0.0], dtype=float)
        points.append(pos.copy())

        rot = rot @ _rot_y(a[4])
        # Joint 6
        pos = pos + rot @ np.array([self.link_wrist_2, 0.0, 0.0], dtype=float)
        points.append(pos.copy())

        return np.asarray(points, dtype=float)

    def update(self, tcp_pose, joint_angles_deg, current_target=None, capture=False):
        now = time.monotonic()
        if now - self._last_draw < self.draw_interval_sec:
            return
        self._last_draw = now

        points = self._forward_kinematics_points(joint_angles_deg)

        if len(tcp_pose) >= 6 and np.linalg.norm(self.tcp_offset_xyz) > 1e-9:
            tcp_xyz = np.array([float(tcp_pose[0]), float(tcp_pose[1]), float(tcp_pose[2])], dtype=float)
            tcp_rot = _rpy_deg_to_rot(tcp_pose[3], tcp_pose[4], tcp_pose[5])
            expected_joint6_xyz = tcp_xyz - tcp_rot @ self.tcp_offset_xyz
            shift_xyz = expected_joint6_xyz - points[-1]
            points = points + shift_xyz

        self.robot_line.set_data(points[:, 0], points[:, 1])
        self.robot_line.set_3d_properties(points[:, 2])

        tcp_xyz = np.array([float(tcp_pose[0]), float(tcp_pose[1]), float(tcp_pose[2])], dtype=float)
        tcp_color = "#d62728" if capture else "#8c564b"
        self.tcp_dot.remove()
        self.tcp_dot = self.ax.scatter([tcp_xyz[0]], [tcp_xyz[1]], [tcp_xyz[2]], s=60, c=tcp_color)

        if current_target is not None:
            target_xyz = np.array([float(current_target[0]), float(current_target[1]), float(current_target[2])], dtype=float)
            self.target_dot.remove()
            self.target_dot = self.ax.scatter([target_xyz[0]], [target_xyz[1]], [target_xyz[2]], s=45, c="#ff7f0e", marker="x")

        self._trace_points.append(tcp_xyz)
        if len(self._trace_points) > 600:
            self._trace_points = self._trace_points[-600:]
        trace = np.asarray(self._trace_points, dtype=float)
        self.trace_line.set_data(trace[:, 0], trace[:, 1])
        self.trace_line.set_3d_properties(trace[:, 2])

        plt.pause(0.001)

    def update_from_arm(self, arm, current_target=None, capture=False):
        pose_code, pose = arm.get_position()
        if pose_code != 0 or pose is None or len(pose) < 3:
            return

        angles_code, angles = arm.get_servo_angle(is_radian=False)
        if angles_code != 0 or angles is None:
            angles = [0.0] * 6

        self.update(pose, angles, current_target=current_target, capture=capture)

    def close(self):
        plt.ioff()
        plt.close(self.fig)