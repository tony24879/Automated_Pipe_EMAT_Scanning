"""Centralized robot hardware and tool configuration constants."""

# Robot controller network address.
ROBOT_IP = "192.168.1.184"

# TCP/tool frame offset: [x, y, z, roll, pitch, yaw].
TCP_OFFSET = [0, 0, 102.6, 0, 0, 0]
#TCP_OFFSET = [0, 0, 90, 0, 0, 0]
#TCP_OFFSET = [0, -40, 77.4, 0, 0, 0]

# Tool payload mass in kilograms.
PAYLOAD_KG = 0.080
#PAYLOAD_KG = 0.094
#PAYLOAD_KG = 0.104

# Tool center of gravity relative to TCP, in millimeters
COG = [0, 0, 75]
#COG = [0, 0, 60]
#COG = [0, -20, 75]