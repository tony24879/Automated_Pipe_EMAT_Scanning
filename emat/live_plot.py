"""Live matplotlib waveform plotting utilities for EMAT data."""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


class LiveWaveformPlot:
    """Realtime line plot for quickly monitoring incoming EMAT signals."""

    def __init__(self):
        """Create a non-blocking matplotlib figure and axis."""
        self._enabled = True
        self._warned = False
        self._min_update_period_s = 0.03
        self._last_draw_ts = 0.0

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [])
        self.peak_scatter = self.ax.scatter([], [], color="red", s=30, zorder=3)

        self.ax.set_title("Real-time EMAT waveform")
        self.ax.set_xlabel("Sample")
        self.ax.set_ylabel("Amplitude")

        # Peak-picking constants are matched to current acquisition settings
        # and should be retuned if sampling rate or transducer setup changes.
        self._skip_samples = 150
        self._min_peak_distance = 100
        self._min_prominence = 500

    def _find_tof_peaks(self, y):
        """Return indices for the first two positive maxima after initial noise."""
        if y.size <= self._skip_samples + 1:
            return []

        post_noise = y[self._skip_samples:]
        peak_indices, _ = find_peaks(post_noise, distance=self._min_peak_distance, prominence=self._min_prominence)

        if len(peak_indices) < 2:
            return []

        first_peak = self._skip_samples + int(peak_indices[0])
        second_peak = self._skip_samples + int(peak_indices[1])
        return [first_peak, second_peak]

    def _disable(self, reason):
        """Disable plotting after a backend/window failure so scans can continue."""
        self._enabled = False
        if not self._warned:
            print(f"Warning: live waveform plot disabled ({reason})")
            self._warned = True

    def update(self, data):
        """Update plotted signal with the newest acquisition buffer."""
        if not self._enabled:
            return False

        if not plt.fignum_exists(self.fig.number):
            self._disable("window was closed")
            return False

        now = time.monotonic()
        if now - self._last_draw_ts < self._min_update_period_s:
            return True

        # Flatten in case the API returns a nested/list-like structure.
        y = np.asarray(data).flatten()

        try:
            self.line.set_xdata(range(len(y)))
            self.line.set_ydata(y)

            peak_indices = self._find_tof_peaks(y)
            if peak_indices:
                peak_points = np.column_stack((peak_indices, y[peak_indices]))
                self.peak_scatter.set_offsets(peak_points)
            else:
                self.peak_scatter.set_offsets(np.empty((0, 2)))

            self.ax.relim()
            self.ax.autoscale_view()

            # Drive the GUI event loop without blocking robot/scan timing.
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self._last_draw_ts = now
            return True
        except Exception as exc:  # noqa: BLE001 - plotting backend failures must not stop acquisition.
            self._disable(str(exc))
            return False

    def close(self):
        """Close the interactive figure and disable interactive mode."""
        self._enabled = False
        try:
            plt.ioff()
            plt.close(self.fig)
        except Exception:  # noqa: BLE001,S110 - ignore backend-specific close failures during shutdown.
            # Ignore close-time backend errors during shutdown.
            pass