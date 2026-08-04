"""Simple motion test that raises the end-effector by 20 mm."""

from config.robot_config import ROBOT_IP
from robot.connection import RobotConnection
from robot.lite6 import Lite6

# Connect and run a tiny Cartesian move as a sanity check.
conn = RobotConnection(ROBOT_IP)

try:
    arm = conn.connect()
    robot = Lite6(arm)

    # Read current robot pose first.
    code, pose = arm.get_position()
    print("Current pose:", pose)

    # Raise end-effector 20 mm in Z while preserving orientation.
    x, y, z, rx, ry, rz = pose

    robot.move_to(
        x,
        y,
        z + 20,
        speed=50
    )

finally:
    # Always disconnect after test.
    conn.disconnect()