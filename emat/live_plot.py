"""Live matplotlib waveform plotting utilities for EMAT data."""

import numpy as np
import matplotlib.pyplot as plt


class LiveWaveformPlot:
    """Realtime line plot for quickly monitoring incoming EMAT signals."""

    def __init__(self):
        """Create a non-blocking matplotlib figure and axis."""
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [])

        self.ax.set_title("Real-time EMAT waveform")
        self.ax.set_xlabel("Sample")
        self.ax.set_ylabel("Amplitude")

    def update(self, data):
        """Update plotted signal with the newest acquisition buffer."""
        # Flatten in case the API returns a nested/list-like structure.
        y = np.asarray(data).flatten()

        self.line.set_xdata(range(len(y)))
        self.line.set_ydata(y)

        self.ax.relim()
        self.ax.autoscale_view()

        plt.pause(0.001)

    def close(self):
        """Close the interactive figure and disable interactive mode."""
        plt.ioff()
        plt.close()