"""Robot setup helpers for TCP and payload configuration."""

import time

from config.robot_config import TCP_OFFSET, PAYLOAD_KG, COG


class RobotSetup:
    """Apply default or custom robot tool/load configuration."""

    def __init__(self, arm):
        """Store initialized xArm API client."""
        self.arm = arm

    def _apply_with_state_retry(self, operation_name, operation):
        """Run one setup API call and retry once if controller reports state-not-ready."""
        code = operation()
        if code == 9:
            self.arm.set_state(0)
            time.sleep(0.1)
            code = operation()

        if code != 0:
            state_code, state = self.arm.get_state()
            err_code, err_warn = self.arm.get_err_warn_code(show=True)
            raise RuntimeError(
                f"{operation_name} failed with code={code}; "
                f"state_query_code={state_code}, state={state}, "
                f"err_query_code={err_code}, err_warn={err_warn}"
            )

        return code

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
        self._apply_with_state_retry(
            "set_tcp_offset",
            lambda: self.arm.set_tcp_offset(tcp_offset),
        )
        print(f"TCP set: {tcp_offset}")

    # Then configure mass properties so motion planning uses correct dynamics.
        self._apply_with_state_retry(
            "set_tcp_load",
            lambda: self.arm.set_tcp_load(payload, center_of_gravity, auto_enable=True),
        )
        print(f"Payload set: {payload} kg, CoG set: {center_of_gravity}")

        print("Robot setup complete ✔")