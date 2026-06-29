"""Robot setup helpers for TCP and payload configuration."""

from config.robot_config import TCP_OFFSET, PAYLOAD_KG, COG


class RobotSetup:
    """Apply default or custom robot tool/load configuration."""

    def __init__(self, arm):
        """Store initialized xArm API client."""
        self.arm = arm

    def configure(self, tcp_offset=None, payload=None, center_of_gravity=None):
        """Configure TCP offset and payload parameters on the controller."""

        if tcp_offset is None:
            tcp_offset = TCP_OFFSET

        if payload is None:
            payload = PAYLOAD_KG

        if center_of_gravity is None:
            center_of_gravity = COG

        print("Configuring robot...")

    # Set the tool center point transform first.
        self.arm.set_tcp_offset(tcp_offset)
        print(f"TCP set: {tcp_offset}")

    # Then configure mass properties so motion planning uses correct dynamics.
        self.arm.set_tcp_load(payload, center_of_gravity, auto_enable=True)
        print(f"Payload set: {payload} kg, CoG set: {center_of_gravity}")

        print("Robot setup complete ✔")