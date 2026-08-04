"""Robot setup helpers for TCP, payload, and collision tool configuration."""

import time

from config.robot_config import (
    COG,
    COLLISION_TOOL_OFFSET_MM,
    COLLISION_TOOL_SIZE_MM,
    COLLISION_TOOL_TYPE,
    PAYLOAD_KG,
    TCP_OFFSET,
)


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

    def configure(
        self,
        tcp_offset=None,
        payload=None,
        center_of_gravity=None,
        collision_tool_type=None,
        collision_tool_size_mm=None,
        collision_tool_offset_mm=None,
    ):
        """Configure TCP offset, payload, and tool collision model on the controller."""

        if tcp_offset is None:
            tcp_offset = TCP_OFFSET

        if payload is None:
            payload = PAYLOAD_KG

        if center_of_gravity is None:
            center_of_gravity = COG

        if collision_tool_type is None:
            collision_tool_type = COLLISION_TOOL_TYPE

        if collision_tool_size_mm is None:
            collision_tool_size_mm = COLLISION_TOOL_SIZE_MM

        if collision_tool_offset_mm is None:
            collision_tool_offset_mm = COLLISION_TOOL_OFFSET_MM

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

        # Tool type 22 expects box dimensions + offset; other types rely on
        # firmware-defined primitive defaults.
        if int(collision_tool_type) == 22:
            x, y, z = [float(v) for v in collision_tool_size_mm]
            x_offset, y_offset, z_offset = [float(v) for v in collision_tool_offset_mm]
            self._apply_with_state_retry(
                "set_collision_tool_model",
                lambda: self.arm.set_collision_tool_model(
                    int(collision_tool_type),
                    x=x,
                    y=y,
                    z=z,
                    x_offset=x_offset,
                    y_offset=y_offset,
                    z_offset=z_offset,
                ),
            )
            print(
                "Collision tool model set: "
                f"type={collision_tool_type}, size_mm={[x, y, z]}, "
                f"offset_mm={[x_offset, y_offset, z_offset]}"
            )
        else:
            self._apply_with_state_retry(
                "set_collision_tool_model",
                lambda: self.arm.set_collision_tool_model(int(collision_tool_type)),
            )
            print(f"Collision tool model set: type={collision_tool_type}")

        print("Robot setup complete ✔")