"""CSV logger for synchronized robot pose and EMAT waveform data."""

import os
import csv
import time


class SyncLogger:
    """Persist time-aligned robot and EMAT samples to disk."""

    def __init__(self, folder="data/raw"):
        """Create timestamped output CSV and write header columns."""
        os.makedirs(folder, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.filename = f"{folder}/sync_scan_{timestamp}.csv"

        self.file = open(self.filename, "w", newline="")
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "t",
            "x", "y", "z",
            "rx", "ry", "rz",
            "signal"
        ])

    def log(self, pose, signal):
        """Append one synchronized row containing pose and waveform."""
        t = time.time()
        x, y, z, rx, ry, rz = pose

        self.writer.writerow([
            t,
            x, y, z,
            rx, ry, rz,
            signal
        ])

    def close(self):
        """Close the open CSV file handle."""
        self.file.close()