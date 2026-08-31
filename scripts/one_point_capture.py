"""Toggle manual mode, let the user jog the arm to a point, then log EMAT captures there."""

import argparse
import time

from config.robot_config import ROBOT_IP
from emat.emat_interface import EMATSession
from emat.live_plot import LiveWaveformPlot
from emat.sync_logger import SyncLogger
from robot.connection import RobotConnection
from robot.lite6 import Lite6
from robot.setup import RobotSetup


def run_one_point_capture(num_captures=1, dwell_seconds=0.5, output_folder="data/raw"):
    """Enable manual mode, wait for user positioning, then log EMAT captures at the rest point."""
    conn = RobotConnection(ROBOT_IP)
    arm = conn.connect()
    robot = Lite6(arm)
    setup = RobotSetup(arm)
    setup.configure()

    try:
        print("Enabling manual mode (teaching mode)...")
        arm.motion_enable(True)
        arm.set_mode(2)
        arm.set_state(0)

        input("Move the arm to the desired point, then press Enter to continue...")

        print("Disabling manual mode...")
        arm.set_mode(0)
        arm.set_state(0)

        pose = robot.get_pose()
        if not pose or len(pose) < 6:
            raise RuntimeError("Unable to read robot pose after manual positioning")
        x, y, z, roll, pitch, yaw = pose[:6]
        print(f"Capture point pose: x={x:.1f}, y={y:.1f}, z={z:.1f}, roll={roll:.1f}, pitch={pitch:.1f}, yaw={yaw:.1f}")

        plotter = None
        try:
            plotter = LiveWaveformPlot()
        except Exception as exc:  # noqa: BLE001 - live plot is optional and must not block capture.
            print(f"Warning: unable to initialize live waveform plot ({exc}); continuing without live plot")

        logger = SyncLogger(folder=output_folder)

        try:
            with EMATSession() as emat:
                print("Configuring EMAT...")
                emat.configure()

                for capture_num in range(num_captures):
                    dwell_until = time.monotonic() + dwell_seconds
                    data = None
                    while time.monotonic() < dwell_until:
                        data = emat.acquire()
                        if plotter is not None:
                            plotter.update(data)
                        time.sleep(0.1)

                    current_pose = robot.get_pose()
                    logger.log(current_pose, data)
                    print(f"Captured {capture_num + 1}/{num_captures}")
        finally:
            if plotter is not None:
                plotter.close()
            logger.close()

        print(f"Logged {num_captures} capture(s) to {logger.filename}")
    finally:
        conn.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manually position the arm and capture EMAT data at that point")
    parser.add_argument("--num-captures", type=int, default=None, help="Number of EMAT captures to log")
    parser.add_argument("--dwell", type=float, default=0.5, help="Seconds to dwell/average per capture")
    parser.add_argument("--output-folder", type=str, default="data/raw", help="Folder for scan logs")
    args = parser.parse_args()

    num_captures = args.num_captures
    if num_captures is None or num_captures < 1:
        num_captures = int(input("Number of EMAT captures: ") or 1)

    run_one_point_capture(
        num_captures=num_captures,
        dwell_seconds=args.dwell,
        output_folder=args.output_folder,
    )
