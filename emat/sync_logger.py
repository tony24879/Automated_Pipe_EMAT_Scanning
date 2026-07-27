"""CSV logger for synchronized robot pose and EMAT waveform data."""

import os
import csv
import time

from scipy.signal import find_peaks


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
        """Compute TOF from the first two detected peaks after initial noise."""
        samples = []
        for value in signal_values:
            try:
                samples.append(float(value))
            except (TypeError, ValueError):
                return ""

        skip_samples = 150
        min_peak_distance = 100
        min_prominence = 500

        if len(samples) <= skip_samples + 1:
            return ""

        post_noise = samples[skip_samples:]

        if not post_noise:
            return ""

        # Only detect local positive maxima (not minima/troughs).
        peak_indices, _ = find_peaks(post_noise, distance=min_peak_distance, prominence=min_prominence)

        if len(peak_indices) < 2:
            return ""

        first_peak_index = skip_samples + int(peak_indices[0])
        second_peak_index = skip_samples + int(peak_indices[1])

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