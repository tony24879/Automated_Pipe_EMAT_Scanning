"""Animate waveform rows from a CSV by scrolling through a row range.

Behavior:
- Reads one CSV file.
- Selects data rows from --start-row to --end-row (both 1-based, inclusive).
- Extracts numeric values from column 11 onward (1-based by default).
- Plots one row waveform at a time and animates by scrolling through rows, using the same
  signal/autocorrelation plot layout and peak-finding logic as plot_row_peaks.py.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

try:
    from plot_row_peaks import (
        compute_autocorrelation,
        draw_signal_and_acf,
        find_first_acf_peak,
        find_two_signal_peaks,
        interpolate_peak_zero_crossing,
    )
except ImportError:
    from data.plot_row_peaks import (
        compute_autocorrelation,
        draw_signal_and_acf,
        find_first_acf_peak,
        find_two_signal_peaks,
        interpolate_peak_zero_crossing,
    )

DEFAULT_SIGNAL_START_COL = 11  # 1-based


def resolve_input_csv_path(input_path: Path) -> Path:
    """Resolve input path with forgiving CSV conventions used in this repo."""
    candidates: list[Path] = []

    candidates.append(input_path)

    if input_path.suffix == "":
        candidates.append(input_path.with_suffix(".csv"))

    if not input_path.is_absolute():
        raw_dir = Path(__file__).resolve().parent / "raw"
        candidates.append(raw_dir / input_path.name)
        if input_path.suffix == "":
            candidates.append(raw_dir / f"{input_path.name}.csv")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate CSV row waveforms by scrolling through a row range."
    )
    parser.add_argument("csv_file", type=Path, help="Path to input CSV file.")
    parser.add_argument(
        "--start-row",
        type=int,
        required=True,
        help="1-based starting row among data rows (inclusive).",
    )
    parser.add_argument(
        "--end-row",
        type=int,
        required=True,
        help="1-based ending row among data rows (inclusive).",
    )
    parser.add_argument(
        "--start-col",
        type=int,
        default=DEFAULT_SIGNAL_START_COL,
        help="1-based first signal column to read (default: 11).",
    )
    parser.add_argument(
        "--header",
        choices=["auto", "yes", "no"],
        default="auto",
        help="Whether CSV has a header row (default: auto).",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=120,
        help="Animation frame interval in milliseconds.",
    )
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="Repeat the animation after the final row.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional animation output path (.gif or .mp4).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the animation window; useful with --save in headless runs.",
    )
    parser.add_argument(
        "--peaks",
        choices=["on", "off"],
        default="on",
        help="Toggle plotting of peak markers on the signal and autocorrelation plots (default: on).",
    )
    return parser.parse_args()


def _is_numeric(text: str) -> bool:
    try:
        float(text.strip())
        return True
    except ValueError:
        return False


def _to_zero_based(index_1_based: int) -> int:
    if index_1_based < 1:
        raise ValueError("Column index must be 1-based and >= 1.")
    return index_1_based - 1


def _has_header(rows: list[list[str]], start_col_idx: int, header_mode: str) -> bool:
    if header_mode == "yes":
        return True
    if header_mode == "no":
        return False

    if not rows:
        return False

    first_row = rows[0]
    if len(first_row) <= start_col_idx:
        return False

    return not _is_numeric(first_row[start_col_idx])


def load_signals_for_row_range(
    csv_path: Path,
    start_row_1_based: int,
    end_row_1_based: int,
    start_col_1_based: int,
    header_mode: str,
) -> tuple[list[int], list[np.ndarray]]:
    if start_row_1_based < 1 or end_row_1_based < 1:
        raise ValueError("Row numbers must be >= 1.")
    if end_row_1_based < start_row_1_based:
        raise ValueError("--end-row must be greater than or equal to --start-row.")

    start_col_idx = _to_zero_based(start_col_1_based)

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError(f"Input CSV is empty: {csv_path}")

    has_header = _has_header(rows, start_col_idx, header_mode)
    data_rows = rows[1:] if has_header else rows

    if end_row_1_based > len(data_rows):
        raise IndexError(
            f"Requested end row {end_row_1_based}, but file has only {len(data_rows)} data rows."
        )

    selected_rows = list(range(start_row_1_based, end_row_1_based + 1))
    signals: list[np.ndarray] = []

    for row_1_based in selected_rows:
        target_row = data_rows[row_1_based - 1]
        if len(target_row) <= start_col_idx:
            raise ValueError(
                f"Row {row_1_based} does not contain column {start_col_1_based}."
            )

        raw_values = target_row[start_col_idx:]
        signal_values: list[float] = []

        for offset, value in enumerate(raw_values, start=start_col_1_based):
            text = value.strip()
            if text == "":
                continue
            try:
                signal_values.append(float(text))
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric value at row {row_1_based}, column {offset}: {value!r}"
                ) from exc

        if not signal_values:
            raise ValueError(
                f"No numeric signal values found for row {row_1_based} from selected start column onward."
            )

        signals.append(np.asarray(signal_values, dtype=float))

    return selected_rows, signals


def animate_row_signals(
    row_numbers: list[int],
    signals: list[np.ndarray],
    csv_name: str,
    interval_ms: int,
    repeat: bool,
    save_path: Path | None,
    show: bool,
    show_peaks: bool = True,
) -> None:
    if not row_numbers or not signals:
        raise ValueError("No rows/signals available for animation.")

    if save_path is None and not show:
        print("No output requested: use --save and/or omit --no-show to render animation.")
        return

    # Precompute peaks/autocorrelation per row up front so each animation frame only redraws.
    frame_data = []
    for y in signals:
        signal_peaks = find_two_signal_peaks(y)
        raw_peak_indices: list[int] = []
        interpolated_peak_index = 0.0
        interpolated_peak_value = 0.0
        interpolated_second_peak_index: float | None = None
        interpolated_second_peak_value: float | None = None

        if signal_peaks:
            raw_peak_index, _, _ = signal_peaks[0]
            raw_peak_indices.append(raw_peak_index)
            interpolated_peak_index, interpolated_peak_value = interpolate_peak_zero_crossing(y, raw_peak_index)

            if len(signal_peaks) > 1:
                second_peak_index, _, _ = signal_peaks[1]
                raw_peak_indices.append(second_peak_index)
                interpolated_second_peak_index, interpolated_second_peak_value = interpolate_peak_zero_crossing(
                    y, second_peak_index
                )

        acf = compute_autocorrelation(y)
        acf_peak_indices = find_first_acf_peak(acf)

        frame_data.append(
            (
                raw_peak_indices,
                interpolated_peak_index,
                interpolated_peak_value,
                interpolated_second_peak_index,
                interpolated_second_peak_value,
                acf,
                acf_peak_indices,
            )
        )

    max_len = max(sig.size for sig in signals)
    y_min = min(float(np.min(sig)) for sig in signals)
    y_max = max(float(np.max(sig)) for sig in signals)
    pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0

    max_acf_len = max(data[5].size for data in frame_data)
    acf_min = min(float(np.min(data[5])) for data in frame_data if data[5].size)
    acf_max = max(float(np.max(data[5])) for data in frame_data if data[5].size)
    acf_pad = 0.05 * (acf_max - acf_min) if acf_max > acf_min else 1.0

    fig, (signal_ax, acf_ax) = plt.subplots(2, 1, figsize=(11, 8), sharex=False)

    def update(frame_idx: int) -> tuple:
        row = row_numbers[frame_idx]
        y = signals[frame_idx]
        (
            raw_peak_indices,
            interpolated_peak_index,
            interpolated_peak_value,
            interpolated_second_peak_index,
            interpolated_second_peak_value,
            acf,
            acf_peak_indices,
        ) = frame_data[frame_idx]

        signal_ax.clear()
        acf_ax.clear()

        title = f"Row {row} waveform from {csv_name} ({frame_idx + 1}/{len(row_numbers)})"
        draw_signal_and_acf(
            signal_ax,
            acf_ax,
            y,
            raw_peak_indices,
            interpolated_peak_index,
            interpolated_peak_value,
            interpolated_second_peak_index,
            interpolated_second_peak_value,
            acf,
            acf_peak_indices,
            title,
            show_peaks=show_peaks,
        )

        signal_ax.set_xlim(0, max_len - 1 if max_len > 0 else 1)
        signal_ax.set_ylim(y_min - pad, y_max + pad)
        acf_ax.set_xlim(0, max_acf_len - 1 if max_acf_len > 0 else 1)
        acf_ax.set_ylim(acf_min - acf_pad, acf_max + acf_pad)

        return ()

    def init() -> tuple:
        return update(0)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(row_numbers),
        init_func=init,
        interval=interval_ms,
        blit=False,
        repeat=repeat,
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = save_path.suffix.lower()
        fps = max(1, int(round(1000 / max(1, interval_ms))))
        if suffix == ".gif":
            animation.save(save_path, writer="pillow", fps=fps)
        else:
            animation.save(save_path, fps=fps)
        print(f"Saved animation to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()

    csv_path = resolve_input_csv_path(args.csv_file)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    row_numbers, signals = load_signals_for_row_range(
        csv_path=csv_path,
        start_row_1_based=args.start_row,
        end_row_1_based=args.end_row,
        start_col_1_based=args.start_col,
        header_mode=args.header,
    )

    animate_row_signals(
        row_numbers=row_numbers,
        signals=signals,
        csv_name=csv_path.name,
        interval_ms=args.interval_ms,
        repeat=args.repeat,
        save_path=args.save,
        show=not args.no_show,
        show_peaks=args.peaks == "on",
    )


if __name__ == "__main__":
    main()
