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
        self._header_written = False
        self._signal_len = None

    def _as_signal_list(self, signal):
        """Normalize waveform input into a flat list of numeric samples."""
        def _flatten(value):
            if isinstance(value, (str, bytes)):
                return [value]

            # Handle numpy arrays (including nested/object arrays) directly.
            if hasattr(value, "ravel") and hasattr(value, "tolist"):
                flattened = value.ravel().tolist()
                if isinstance(flattened, list):
                    return flattened
                return [flattened]

            try:
                iterator = iter(value)
            except TypeError:
                return [value]

            flattened = []
            for item in iterator:
                flattened.extend(_flatten(item))
            return flattened

        return _flatten(signal)

    def _ensure_header(self, signal_values):
        """Write CSV header once, including one column per signal sample."""
        if self._header_written:
            return

        self._signal_len = len(signal_values)
        header = [
            "t",
            "x", "y", "z",
            "rx", "ry", "rz",
            "Theta (deg)",
            "Axis Position (mm)",
            "Time of Flight (s)",
            *[f"signal_{i}" for i in range(self._signal_len)],
        ]
        self.writer.writerow(header)
        self._header_written = True

    def _compute_time_of_flight(self, signal_values):
        """Compute TOF from peak index difference in two fixed sample windows."""
        samples = []
        for value in signal_values:
            try:
                samples.append(float(value))
            except (TypeError, ValueError):
                return ""

        # Need at least through index 690 for the second peak window.
        if len(samples) <= 690:
            return ""

        first_start, first_end = 300, 380
        second_start, second_end = 610, 690

        first_window = samples[first_start:first_end + 1]
        second_window = samples[second_start:second_end + 1]

        if not first_window or not second_window:
            return ""

        first_peak_index = first_start + max(range(len(first_window)), key=first_window.__getitem__)
        second_peak_index = second_start + max(range(len(second_window)), key=second_window.__getitem__)

        sample_difference = second_peak_index - first_peak_index
        return sample_difference * (20e-9)

    def log(self, pose, signal, theta="", axis_position=""):
        """Append one synchronized row containing pose and waveform."""
        t = time.time()
        x, y, z, rx, ry, rz = pose
        signal_values = self._as_signal_list(signal)
        time_of_flight = self._compute_time_of_flight(signal_values)

        self._ensure_header(signal_values)

        # Keep column count stable if later samples have different lengths.
        if len(signal_values) < self._signal_len:
            signal_values = signal_values + [""] * (self._signal_len - len(signal_values))
        elif len(signal_values) > self._signal_len:
            signal_values = signal_values[:self._signal_len]

        self.writer.writerow([
            t,
            x, y, z,
            rx, ry, rz,
            theta,
            axis_position,
            time_of_flight,
            *signal_values
        ])

    def close(self):
        """Close the open CSV file handle."""
        self.file.close()