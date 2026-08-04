"""Small synchronized robot + EMAT scan test over a short point list."""

import time

from config.robot_config import ROBOT_IP
from emat.emat_interface import EMATSession
from emat.live_plot import LiveWaveformPlot
from emat.sync_logger import SyncLogger
from robot.connection import RobotConnection
from robot.lite6 import Lite6
from robot.setup import RobotSetup

# ------------------------
# ROBOT SETUP
# ------------------------
# Connect and apply configured tool/load parameters.
conn = RobotConnection(ROBOT_IP)
arm = conn.connect()

robot = Lite6(arm)

setup = RobotSetup(arm)
setup.configure()

# ------------------------
# SMALL TEST GRID
# ------------------------
points = [
    (250, 0, 150),
    (250, 20, 150),
    (250, 40, 150),
    (250, 60, 150),
]

# ------------------------
# SYNCHRONISED ACQUISITION
# ------------------------
# Set up logging and plotting before entering acquisition loop.
logger = SyncLogger()
plotter = LiveWaveformPlot()

try:
    with EMATSession() as emat:

        print("Configuring EMAT...")
        emat.configure()

        print("Starting scan...")

        for (x, y, z) in points:

            # 1) Move robot to next test location.
            robot.move_to(x, y, z, speed=40)

            # 2) Dwell at position while continuously updating waveform view.
            dwell_until = time.monotonic() + 5
            data = None
            while time.monotonic() < dwell_until:
                data = emat.acquire()
                plotter.update(data)
                time.sleep(0.1)

            # 3) Read current robot pose after settling.
            pose = robot.get_pose()

            # 4) Log synchronized pose + EMAT signal sample.
            logger.log(pose, data)

            print(f"Captured at ({x},{y},{z})")

finally:
    # Ensure all resources are closed on normal exit or error.
    plotter.close()
    logger.close()
    conn.disconnect()