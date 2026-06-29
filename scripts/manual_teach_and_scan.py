"""Record a manually taught robot trajectory and replay it with EMAT scanning."""

import time

from robot.connection import RobotConnection
from robot.lite6 import Lite6

from emat.emat_interface import EMATSession
from emat.live_plot import LiveWaveformPlot
from emat.sync_logger import SyncLogger

from config.robot_config import ROBOT_IP


def _make_trajectory_name():
    """Create a unique trajectory name using the current timestamp."""
    return f"manualscan{time.strftime('%Y%m%d%H%M%S')}"


def _get_trajectory_duration(arm, trajectory_name):
    """Look up a saved trajectory duration for replay timing."""
    result = arm.get_trajectories()
    if isinstance(result, tuple) and len(result) == 2:
        code, trajectories = result
        if code != 0:
            return None
    else:
        trajectories = result

    for trajectory in trajectories or []:
        name = str(trajectory.get("name", ""))
        if name == trajectory_name or name == f"{trajectory_name}.traj" or name.endswith(f"{trajectory_name}.traj"):
            return float(trajectory.get("duration", 0))
    return None


def main():
    """Run interactive teach-record-playback workflow while logging EMAT data."""
    conn = RobotConnection(ROBOT_IP)
    arm = conn.connect()
    robot = Lite6(arm)
    plotter = LiveWaveformPlot()
    logger = SyncLogger()
    trajectory_name = _make_trajectory_name()

    try:
        # Enable teach mode for manual hand-guided recording.
        print("Switching robot to teach mode for manual recording...")
        arm.motion_enable(True)
        arm.set_mode(2)
        arm.set_state(0)

        input(
            "Press Enter to start recording. Then hold the side teach button and move the robot by hand."
        )

        record_code = arm.start_record_trajectory()
        if record_code != 0:
            raise RuntimeError(f"Failed to start trajectory recording: code={record_code}")

        input(
            "Move the robot manually now. Press Enter when you want to stop recording and save the trajectory."
        )

        stop_code = arm.stop_record_trajectory(filename=trajectory_name)
        if stop_code != 0:
            raise RuntimeError(f"Failed to stop/save trajectory: code={stop_code}")

        arm.set_mode(0)
        arm.set_state(0)

        duration = _get_trajectory_duration(arm, trajectory_name)
        if duration is None:
            print(f"Trajectory saved as {trajectory_name}.traj")
            duration = 5.0
        else:
            print(f"Trajectory saved as {trajectory_name}.traj, duration ~ {duration:.2f}s")

        with EMATSession() as emat:
            print("Configuring EMAT...")
            emat.configure()
            print("Replaying recorded trajectory while scanning EMAT...")

            play_code = arm.playback_trajectory(filename=trajectory_name, wait=False)
            if play_code != 0:
                raise RuntimeError(f"Failed to start trajectory playback: code={play_code}")

            start_deadline = time.monotonic() + 5
            while time.monotonic() < start_deadline and not arm.get_is_moving():
                time.sleep(0.05)

            if not arm.get_is_moving():
                raise RuntimeError("Trajectory playback did not start moving.")

            # Continue acquiring until expected duration elapses and motion has fully stopped.
            end_time = time.monotonic() + duration + 1.0
            while time.monotonic() < end_time or arm.get_is_moving():
                data = emat.acquire()
                pose = robot.get_pose()
                plotter.update(data)
                logger.log(pose, data)
                time.sleep(0.05)

            print("Playback scan complete")

    finally:
        # Ensure resources are closed even if runtime exceptions occur.
        plotter.close()
        logger.close()
        conn.disconnect()


if __name__ == "__main__":
    main()
