"""Robot connection helpers for initializing and shutting down xArm sessions."""


from xarm.wrapper import XArmAPI


class RobotConnection:
    """Create and manage a connection to the UFACTORY controller."""

    def __init__(self, ip):
        """Store target robot IP and clear any previous client handle."""
        self.ip = ip
        self.arm = None

    def connect(self):
        """Connect to the robot, initialize state, and verify readiness."""
        print(f"Connecting to robot at {self.ip}...")
        self.arm = XArmAPI(self.ip)

        motion_code = self.arm.motion_enable(True)
        mode_code = self.arm.set_mode(0)
        state_code = self.arm.set_state(0)

        if motion_code != 0 or mode_code != 0 or state_code != 0:
            raise RuntimeError(
                f"Robot setup failed: motion_enable={motion_code}, set_mode={mode_code}, set_state={state_code}"
            )

        print("Running reset to wake and home the robot...")
        self.arm.reset(wait=True)

    # Confirm controller status after reset before continuing.
        last_state_code, last_state = self.arm.get_state()
        last_err_code, last_err_warn = self.arm.get_err_warn_code(show=True)
        if last_state_code != 0 or last_err_code != 0 or last_err_warn != [0, 0]:
            raise RuntimeError(
                f"Robot did not become ready after reset: state={last_state}, err_warn={last_err_warn}"
            )

        if last_state == 2:
            print("Robot is sleeping after reset; motion commands will wake it")
        elif last_state != 0:
            print(f"Robot connected with state {last_state}")

        print("Robot connected ✔")
        return self.arm

    def disconnect(self):
        """Close the controller connection if it exists."""
        if self.arm:
            self.arm.disconnect()
            print("Robot disconnected")