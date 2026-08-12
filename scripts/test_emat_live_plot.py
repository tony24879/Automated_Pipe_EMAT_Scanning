"""Connect to EMAT and show a live waveform plot."""

import csv
import msvcrt
import os
import time

from emat.emat_interface import EMATSession
from emat.live_plot import LiveWaveformPlot


def _signal_to_row(signal):
    """Normalize waveform-like input into a flat CSV row list."""
    if hasattr(signal, "ravel") and hasattr(signal, "tolist"):
        values = signal.ravel().tolist()
        return values if isinstance(values, list) else [values]

    try:
        iterator = iter(signal)
    except TypeError:
        return [signal]

    row = []
    for item in iterator:
        if isinstance(item, (str, bytes)):
            row.append(item)
            continue

        try:
            nested = iter(item)
        except TypeError:
            row.append(item)
            continue

        row.extend(list(nested))
    return row


def main():
    """Acquire continuously from EMAT and update a realtime plot."""
    plotter = LiveWaveformPlot()
    csv_path = os.path.join("data", "raw", "live_plot_arrays.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    saved_count = 0

    try:
        with EMATSession() as emat, open(csv_path, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            print("Configuring EMAT...")
            emat.configure()
            print("Streaming live waveform. Press Enter to save current array, Ctrl+C to stop.")

            while True:
                data = emat.acquire()
                plotter.update(data)

                # Capture Enter without blocking acquisition/plot updates.
                should_save = False
                while msvcrt.kbhit():
                    key = msvcrt.getwch()
                    if key in ("\r", "\n"):
                        should_save = True

                if should_save:
                    writer.writerow(_signal_to_row(data))
                    csv_file.flush()
                    saved_count += 1
                    print(f"Saved array #{saved_count} to {csv_path}")

                time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nStopped live plot.")

    finally:
        plotter.close()


if __name__ == "__main__":
    main()
