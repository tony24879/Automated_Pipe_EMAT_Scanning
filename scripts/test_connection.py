"""Minimal script to verify controller connectivity."""

from robot.connection import RobotConnection
from config.robot_config import ROBOT_IP

# Create and open robot connection.
conn = RobotConnection(ROBOT_IP)
arm = conn.connect()

print("Connected!")

# Print firmware/API version reported by controller.
code, version = arm.get_version()
print(version)

# Disconnect cleanly after test.
conn.disconnect()