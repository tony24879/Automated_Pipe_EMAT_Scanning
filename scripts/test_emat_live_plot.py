"""Connect to EMAT and show a live waveform plot."""

import time

from emat.emat_interface import EMATSession
from emat.live_plot import LiveWaveformPlot


def main():
    """Acquire continuously from EMAT and update a realtime plot."""
    plotter = LiveWaveformPlot()

    try:
        with EMATSession() as emat:
            print("Configuring EMAT...")
            emat.configure()
            print("Streaming live waveform. Press Ctrl+C to stop.")

            while True:
                data = emat.acquire()
                plotter.update(data)
                time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nStopped live plot.")

    finally:
        plotter.close()


if __name__ == "__main__":
    main()
