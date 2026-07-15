"""Centralized robot hardware and tool configuration constants."""

# Robot controller network address.
ROBOT_IP = "192.168.1.184"

# TCP/tool frame offset: [x, y, z, roll, pitch, yaw].
TCP_OFFSET = [0, 0, 104.0, 0, 0, 90]
#TCP_OFFSET = [0, 0, 102.6, 0, 0, 0]
#TCP_OFFSET = [0, 0, 90, 0, 0, 0]
#TCP_OFFSET = [0, -40, 77.4, 0, 0, 0]

# Tool payload mass in kilograms.
PAYLOAD_KG = 0.090
#PAYLOAD_KG = 0.080
#PAYLOAD_KG = 0.094
#PAYLOAD_KG = 0.104

# Tool center of gravity relative to TCP, in millimeters
COG = [0, 0, 80]
#COG = [0, 0, 75]
#COG = [0, 0, 60]
#COG = [0, -20, 75]

# Collision model for end effector self-collision checking.
# xArm primitive type 22 = cuboid.
COLLISION_TOOL_TYPE = 22
COLLISION_TOOL_SIZE_MM = [38.0, 38.0, 113.0]  # [x, y, z]
COLLISION_TOOL_OFFSET_MM = [0.0, 0.0, 0.0]    # [x_offset, y_offset, z_offset]