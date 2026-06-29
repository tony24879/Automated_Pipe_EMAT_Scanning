"""High-level motion helpers for the Lite6 robot arm."""

import math
import time


class Lite6:
    """Convenience wrapper around xArm position/state APIs."""

    def __init__(self, arm):
        """Store the initialized xArm API client."""
        self.arm = arm

    # -------------------------
    # BASIC MOTION
    # -------------------------

    def move_to(self, x, y, z, speed=50, wait=True, roll=None, pitch=None, yaw=None):
        """Move to an absolute Cartesian pose while preserving orientation when omitted."""
        if roll is None or pitch is None or yaw is None:
            code, pose = self.arm.get_position()
            if code == 0 and pose and len(pose) >= 6 and all(math.isfinite(value) for value in pose[3:6]):
                roll, pitch, yaw = pose[3:6]
            else:
                roll, pitch, yaw = 0, 0, 0

        return self.arm.set_position(
            x,
            y,
            z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            speed=speed,
            wait=wait,
            auto_enable=True,
        )

    def move_relative(self, dx=0, dy=0, dz=0, speed=50):
        """Move by a Cartesian delta from the current pose."""
        code, pos = self.arm.get_position()
        x, y, z, rx, ry, rz = pos

        return self.move_to(
            x + dx,
            y + dy,
            z + dz,
            speed=speed
        )

    def get_pose(self):
        """Read and return the current Cartesian pose."""
        code, pose = self.arm.get_position()
        return pose

    # -------------------------
    # SAFETY LAYER
    # -------------------------

    def stop(self):
        """Trigger emergency stop state."""
        self.arm.set_state(4)  # emergency stop mode

    def relax(self):
        """Return robot to normal operational state."""
        self.arm.set_state(0)

    # -------------------------
    # HOMING / RESET
    # -------------------------

    def home(self):
        """Move to a simple predefined home position."""
        self.move_to(300, 0, 200, speed=30)

    # -------------------------
    # SCANNING HELPERS
    # -------------------------

    def move_and_settle(self, x, y, z, delay=0.1):
        """Move to target pose and wait briefly for mechanical settling."""
        self.move_to(x, y, z)
        time.sleep(delay)
        return self.get_pose()