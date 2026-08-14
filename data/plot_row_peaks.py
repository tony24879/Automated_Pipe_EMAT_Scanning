"""Extract one or more row waveforms from CSV, find two TOF peaks, and plot results.

Behavior:
- Reads one CSV file.
- Selects one or more user-specified rows (1-based, data rows by default).
- Extracts values from column 11 onward (1-based).
- Applies the same peak-finding settings used in sync_logger.py/live_plot.py.
- Prints the two peak indices and plots waveform + peak markers for each row.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# Match existing TOF peak logic in emat/sync_logger.py and emat/live_plot.py.
SKIP_SAMPLES = 200 #150
MIN_PEAK_DISTANCE = 250 #280
MIN_PROMINENCE = 30 #500
DEFAULT_SIGNAL_START_COL = 11  # 1-based


def resolve_input_csv_path(input_path: Path) -> Path:
    """Resolve input path with forgiving CSV conventions used in this repo."""
    candidates: list[Path] = []

    # Candidate 1: exactly what user passed.
    candidates.append(input_path)

    # Candidate 2: append .csv if missing extension.
    if input_path.suffix == "":
        candidates.append(input_path.with_suffix(".csv"))

    # Candidate 3/4: if relative, also try under data/raw.
    if not input_path.is_absolute():
        raw_dir = Path(__file__).resolve().parent / "raw"
        candidates.append(raw_dir / input_path.name)
        if input_path.suffix == "":
            candidates.append(raw_dir / f"{input_path.name}.csv")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Return first candidate to preserve a clear not-found message.
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find and plot first two TOF peaks from one or more CSV row waveforms."
    )
    parser.add_argument(
        "csv_file",
        type=Path,
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--row",
        type=int,
        required=False,
        help="1-based row number among data rows (ignores header when present).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=None,
        help="Space-separated list of 1-based row numbers among data rows.",
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
        "--save",
        type=Path,
        default=None,
        help="Optional output image path. If omitted, figure is only shown.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a figure window; useful for headless runs.",
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
        # If first row is too short, default to no header and let validation fail later.
        return False

    return not _is_numeric(first_row[start_col_idx])


def load_signal_row(
    csv_path: Path,
    row_1_based: int,
    start_col_1_based: int,
    header_mode: str,
) -> np.ndarray:
    if row_1_based < 1:
        raise ValueError("--row must be >= 1.")

    start_col_idx = _to_zero_based(start_col_1_based)

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError(f"Input CSV is empty: {csv_path}")

    has_header = _has_header(rows, start_col_idx, header_mode)
    data_rows = rows[1:] if has_header else rows

    if row_1_based > len(data_rows):
        raise IndexError(
            f"Requested row {row_1_based}, but file has only {len(data_rows)} data rows."
        )

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
            # Skip trailing or sparse empty cells.
            continue
        try:
            signal_values.append(float(text))
        except ValueError as exc:
            raise ValueError(
                f"Non-numeric value at selected row, column {offset}: {value!r}"
            ) from exc

    if not signal_values:
        raise ValueError(
            "No numeric signal values found from selected start column onward."
        )

    return np.asarray(signal_values, dtype=float)


def find_two_signal_peaks(y: np.ndarray) -> list[tuple[int, str, float]]:
    """Return the first two raw-signal peaks while keeping each peak on the same polarity branch.

    We search both the signal and its flipped version, compare the first peak in each branch,
    and then only consider the same branch for the second peak. This prevents mixing a
    positive peak with a negative peak when picking the pair.
    """
    if y.size <= SKIP_SAMPLES + 1:
        return []

    post_noise = y[SKIP_SAMPLES:]
    pos_peak_indices, _ = find_peaks(
        post_noise,
        distance=MIN_PEAK_DISTANCE,
        prominence=MIN_PROMINENCE,
    )
    neg_peak_indices, _ = find_peaks(
        -post_noise,
        distance=MIN_PEAK_DISTANCE,
        prominence=MIN_PROMINENCE,
    )

    positive_peaks = [
        (SKIP_SAMPLES + int(idx), "normal", float(y[SKIP_SAMPLES + int(idx)]))
        for idx in pos_peak_indices
    ]
    negative_peaks = [
        (SKIP_SAMPLES + int(idx), "flipped", float(y[SKIP_SAMPLES + int(idx)]))
        for idx in neg_peak_indices
    ]

    if not positive_peaks and not negative_peaks:
        return []

    first_candidates = []
    if positive_peaks:
        first_candidates.append(positive_peaks[0])
    if negative_peaks:
        first_candidates.append(negative_peaks[0])

    first_peak_index, first_source, first_peak_value = max(
        first_candidates,
        key=lambda item: abs(item[2]),
    )

    selected: list[tuple[int, str, float]] = [(first_peak_index, first_source, first_peak_value)]
    same_polarity_peaks = positive_peaks if first_source == "normal" else negative_peaks
    if len(same_polarity_peaks) > 1:
        second_peak = max(
            same_polarity_peaks[1:],
            key=lambda item: abs(item[2]),
        )
        selected.append((second_peak[0], second_peak[1], second_peak[2]))

    return selected[:2]


def interpolate_peak_zero_crossing(y: np.ndarray, peak_idx: int) -> tuple[float, float]:
    """Estimate a sub-sample peak location by fitting a parabola to the three samples
    around the peak and finding the vertex. This is the usual quadratic/zero-crossing
    interpolation used when the true peak lies between integer samples.
    """
    if peak_idx <= 0 or peak_idx >= y.size - 1:
        return float(peak_idx), float(y[peak_idx])

    y_minus = float(y[peak_idx - 1])
    y0 = float(y[peak_idx])
    y_plus = float(y[peak_idx + 1])
    denominator = y_minus - 2.0 * y0 + y_plus

    if abs(denominator) < 1e-12:
        return float(peak_idx), y0

    delta = 0.5 * (y_minus - y_plus) / denominator
    interpolated_x = float(peak_idx) + delta

    interpolated_y = (
        y0
        + 0.5 * (y_plus - y_minus) * delta
        + 0.5 * (y_minus - 2.0 * y0 + y_plus) * delta**2
    )
    return interpolated_x, float(interpolated_y)


def compute_autocorrelation(y: np.ndarray) -> np.ndarray:
    post_noise = y[SKIP_SAMPLES:] if y.size > SKIP_SAMPLES else y
    if post_noise.size == 0:
        return np.asarray([], dtype=float)

    acf = np.correlate(post_noise, post_noise, mode="full")
    return acf[acf.size // 2 :]


def find_two_acf_peaks(acf: np.ndarray) -> list[int]:
    if acf.size <= 2:
        return []

    acf_search = acf[1:]
    peak_indices, _ = find_peaks(
        acf_search,
        distance=MIN_PEAK_DISTANCE,
        prominence=MIN_PROMINENCE,
    )

    if len(peak_indices) < 2:
        return []

    first_peak = 1 + int(peak_indices[0])
    second_peak = 1 + int(peak_indices[1])
    return [first_peak, second_peak]


def normalize_rows(row: int | None, rows: list[int] | None) -> list[int]:
    selected_rows: list[int] = []
    if row is not None:
        selected_rows.append(row)
    if rows:
        selected_rows.extend(rows)

    if not selected_rows:
        raise ValueError("Provide at least one row using --row or --rows.")

    unique_rows: list[int] = []
    seen: set[int] = set()
    for r in selected_rows:
        if r < 1:
            raise ValueError("Row numbers must be >= 1.")
        if r not in seen:
            seen.add(r)
            unique_rows.append(r)
    return unique_rows


def build_save_path(base_save_path: Path, row: int, row_count: int) -> Path:
    if row_count == 1:
        return base_save_path

    stem = base_save_path.stem
    suffix = base_save_path.suffix or ".png"
    parent = base_save_path.parent
    return parent / f"{stem}_row{row}{suffix}"


def plot_signal_with_peaks(
    y: np.ndarray,
    raw_peak_indices: list[int],
    interpolated_peak_index: float,
    interpolated_peak_value: float,
    interpolated_second_peak_index: float | None,
    interpolated_second_peak_value: float | None,
    acf: np.ndarray,
    acf_peak_indices: list[int],
    title: str,
    save_path: Path | None,
    show: bool,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    signal_ax, acf_ax = axes

    x_signal = np.arange(y.size)
    signal_ax.plot(x_signal, y, label="Signal", color="tab:blue")

    if raw_peak_indices:
        raw_peak_array = np.array(raw_peak_indices, dtype=int)
        signal_ax.scatter(
            raw_peak_array,
            y[raw_peak_array],
            color="red",
            s=50,
            zorder=3,
            label="Raw signal peaks",
        )
        for idx in raw_peak_array:
            signal_ax.axvline(idx, color="red", linestyle="--", alpha=0.5)

    signal_ax.axvline(
        interpolated_peak_index,
        color="tab:orange",
        linestyle=":",
        alpha=0.8,
        label="Interpolated peak 1",
    )
    signal_ax.scatter(
        interpolated_peak_index,
        interpolated_peak_value,
        color="tab:orange",
        s=50,
        zorder=3,
        label="Peak 1 after interpolation",
    )

    if interpolated_second_peak_index is not None and interpolated_second_peak_value is not None:
        signal_ax.axvline(
            interpolated_second_peak_index,
            color="tab:purple",
            linestyle=":",
            alpha=0.8,
            label="Interpolated peak 2",
        )
        signal_ax.scatter(
            interpolated_second_peak_index,
            interpolated_second_peak_value,
            color="tab:purple",
            s=50,
            zorder=3,
            label="Peak 2 after interpolation",
        )

    signal_ax.set_title(f"{title} - Raw Signal")
    signal_ax.set_xlabel("Sample")
    signal_ax.set_ylabel("Amplitude")
    signal_ax.grid(alpha=0.25)
    signal_ax.legend(loc="best")

    x_acf = np.arange(acf.size)
    acf_ax.plot(x_acf, acf, label="Autocorrelation", color="tab:green")
    if acf_peak_indices:
        acf_peaks_np = np.array(acf_peak_indices, dtype=int)
        acf_ax.scatter(
            acf_peaks_np,
            acf[acf_peaks_np],
            color="red",
            s=40,
            zorder=3,
            label="ACF peaks",
        )
    acf_ax.set_title(f"{title} - Autocorrelation")
    acf_ax.set_xlabel("Lag")
    acf_ax.set_ylabel("Correlation")
    acf_ax.grid(alpha=0.25)
    acf_ax.legend(loc="best")

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved plot image to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()

    csv_path = resolve_input_csv_path(args.csv_file)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    rows = normalize_rows(args.row, args.rows)

    for row in rows:
        signal = load_signal_row(
            csv_path=csv_path,
            row_1_based=row,
            start_col_1_based=args.start_col,
            header_mode=args.header,
        )

        signal_peaks = find_two_signal_peaks(signal)
        if len(signal_peaks) < 2:
            print(f"\n=== Row {row} ===")
            print(
                "Fewer than two usable raw-signal peaks found with settings: "
                f"skip={SKIP_SAMPLES}, distance={MIN_PEAK_DISTANCE}, prominence={MIN_PROMINENCE}."
            )
            continue

        raw_peak_index, peak_source, peak_value = signal_peaks[0]
        second_peak_index, second_source, second_peak_value = signal_peaks[1]
        interpolated_peak_index, interpolated_peak_value = interpolate_peak_zero_crossing(
            signal,
            raw_peak_index,
        )
        interpolated_second_peak_index, interpolated_second_peak_value = interpolate_peak_zero_crossing(
            signal,
            second_peak_index,
        )
        acf = compute_autocorrelation(signal)
        acf_peak_indices = find_two_acf_peaks(acf)

        print(f"\n=== Row {row} ===")
        print(
            "Raw signal peak 1 from "
            f"{peak_source} signal: index={raw_peak_index}, amplitude={peak_value:.6f}"
        )
        print(
            "Raw signal peak 2 from "
            f"{second_source} signal: index={second_peak_index}, amplitude={second_peak_value:.6f}"
        )
        print(
            "Zero-crossing interpolation around peak 1: "
            f"index_before={raw_peak_index}, value_before={float(signal[raw_peak_index]):.6f}; "
            f"index_after={interpolated_peak_index:.6f}, value_after={interpolated_peak_value:.6f}"
        )
        print(
            "Zero-crossing interpolation around peak 2: "
            f"index_before={second_peak_index}, value_before={float(signal[second_peak_index]):.6f}; "
            f"index_after={interpolated_second_peak_index:.6f}, value_after={interpolated_second_peak_value:.6f}"
        )

        if len(acf_peak_indices) < 2:
            print(
                "Fewer than two peaks found in autocorrelation with settings: "
                f"skip={SKIP_SAMPLES}, distance={MIN_PEAK_DISTANCE}, prominence={MIN_PROMINENCE}."
            )
        else:
            print(f"Autocorrelation peak 1 lag index: {acf_peak_indices[0]}")
            print(f"Autocorrelation peak 2 lag index: {acf_peak_indices[1]}")

        print(f"Signal length: {len(signal)}")

        title = f"Row {row} waveform from {csv_path.name}"
        save_path = (
            build_save_path(args.save, row=row, row_count=len(rows))
            if args.save is not None
            else None
        )
        plot_signal_with_peaks(
            signal,
            [raw_peak_index, second_peak_index],
            interpolated_peak_index,
            interpolated_peak_value,
            interpolated_second_peak_index,
            interpolated_second_peak_value,
            acf,
            acf_peak_indices,
            title=title,
            save_path=save_path,
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()
