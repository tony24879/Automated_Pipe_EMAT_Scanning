"""Live matplotlib waveform plotting utilities for EMAT data."""

import time

import numpy as np
import matplotlib.pyplot as plt


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

        self.ax.set_title("Real-time EMAT waveform")
        self.ax.set_xlabel("Sample")
        self.ax.set_ylabel("Amplitude")

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

            self.ax.relim()
            self.ax.autoscale_view()

            # Drive the GUI event loop without blocking robot/scan timing.
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self._last_draw_ts = now
            return True
        except Exception as exc:
            self._disable(str(exc))
            return False

    def close(self):
        """Close the interactive figure and disable interactive mode."""
        self._enabled = False
        try:
            plt.ioff()
            plt.close(self.fig)
        except Exception:
            # Ignore close-time backend errors during shutdown.
            pass