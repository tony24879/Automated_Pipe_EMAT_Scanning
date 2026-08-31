"""Calculate repeatability metrics from grouped scan values.

Grouping behavior:
- Read rows with numeric values in column 10 (1-based).
- Start a new group when column 8 or 9 changes compared to the previous usable row.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from repeat_point_grouping import (
        grouped_tof_values,
        group_rows_by_repeat_point,
        group_row_indices_by_repeat_point,
    )
except ModuleNotFoundError:
    from data.repeat_point_grouping import (
        grouped_tof_values,
        group_rows_by_repeat_point,
        group_row_indices_by_repeat_point,
    )

RAW_DIR = Path(__file__).resolve().parent / "raw"
TARGET_TOF_COL = 10


def read_override_values(path: Path) -> list[float]:
    """Read numeric values separated by newlines, commas, or spaces."""
    values: list[float] = []
    with path.open("r", encoding="utf-8") as source_file:
        for raw_line in source_file:
            for token in raw_line.replace(",", " ").split():
                if token.strip() == "":
                    continue
                try:
                    values.append(float(token))
                except ValueError:
                    continue
    return values


def transform_group_value(
    value: float,
    source_mode: str,
    estimate: float | None = None,
    comparison_value: float | None = None,
) -> float:
    if source_mode == "ToF":
        return value

    if source_mode == "Thickness":
        if estimate is None:
            raise ValueError("Thickness source requires an estimate value.")
        return (value / 2.0) * estimate

    if source_mode == "Speed":
        if estimate is None:
            raise ValueError("Speed source requires an estimate value.")
        if value == 0:
            raise ValueError("Speed conversion requires a non-zero TOF value.")
        return (estimate * 2.0) / value

    if source_mode == "Errors":
        if comparison_value is None:
            raise ValueError("Errors source requires a comparison value.")
        if comparison_value == 0:
            raise ValueError("Errors conversion requires a non-zero comparison value.")
        return value / comparison_value

    raise ValueError(f"Unsupported value source: {source_mode}")


def resolve_input_path(raw_file: str) -> Path:
    candidate = Path(raw_file)
    if candidate.is_absolute():
        return candidate
    return RAW_DIR / candidate


def get_rows(csv_path: Path) -> list[list[str]]:
    import csv

    with csv_path.open("r", newline="", encoding="utf-8") as source_file:
        reader = csv.reader(source_file)
        return [row for row in reader if row]


def _split_override_by_group_indices(override_values: list[float], group_indices: list[list[int]]) -> list[list[float]]:
    expected_count = sum(len(indices) for indices in group_indices)
    if len(override_values) != expected_count:
        raise ValueError(
            "Override values length does not match grouped row count. "
            f"Expected {expected_count}, got {len(override_values)}."
        )

    return [[float(override_values[i]) for i in indices] for indices in group_indices]


def _group_row_indices(rows: list[list[str]]) -> list[list[int]]:
    return group_row_indices_by_repeat_point(rows, tof_col=TARGET_TOF_COL, key_cols=(8, 9), skip_header=True)


def _grouped_tof(rows: list[list[str]]) -> list[list[float]]:
    grouped_rows = group_rows_by_repeat_point(rows, tof_col=TARGET_TOF_COL, key_cols=(8, 9), skip_header=True)
    groups = grouped_tof_values(grouped_rows, tof_col=TARGET_TOF_COL)
    if not groups:
        raise ValueError("No grouped numeric TOF values were found.")
    return groups


def group_metric(group: list[float], method: str) -> float:
    """Return the chosen repeatability metric for one group."""
    if method == "range":
        return float(np.max(group) - np.min(group))
    if method in {"std", "standard_deviation", "standard deviation"}:
        return float(np.std(group))
    raise ValueError("method must be one of: 'range' or 'std'.")


def _build_groups_for_source(
    rows: list[list[str]],
    source_mode: str,
    estimate: float | None = None,
    second_rows: list[list[str]] | None = None,
    override_values: list[float] | None = None,
) -> list[list[float]]:
    if override_values is not None and source_mode != "ToF":
        raise ValueError("Override values are only supported with source mode ToF.")

    if override_values is not None:
        group_indices = _group_row_indices(rows)
        if not group_indices:
            raise ValueError("No grouped numeric TOF rows were found for override values.")
        return _split_override_by_group_indices(override_values, group_indices)

    groups = _grouped_tof(rows)
    if source_mode == "ToF":
        return groups

    if source_mode == "Errors":
        if second_rows is None:
            raise ValueError("Errors source mode requires a second CSV.")
        second_groups = _grouped_tof(second_rows)
        if len(second_groups) != len(groups):
            raise ValueError(
                "Input CSV and comparison CSV produced different group counts. "
                f"Got {len(groups)} and {len(second_groups)}."
            )

        transformed_groups: list[list[float]] = []
        for first_group, second_group in zip(groups, second_groups):
            if len(first_group) != len(second_group):
                raise ValueError("Group lengths differ between input CSV and comparison CSV.")
            transformed_groups.append(
                [
                    transform_group_value(
                        value,
                        source_mode=source_mode,
                        estimate=estimate,
                        comparison_value=comparison,
                    )
                    for value, comparison in zip(first_group, second_group)
                ]
            )
        return transformed_groups

    return [
        [transform_group_value(value, source_mode=source_mode, estimate=estimate) for value in group]
        for group in groups
    ]


def calculate_repeatability_error(
    input_csv: Path,
    method: str,
    override_values: list[float] | None = None,
    source_mode: str = "ToF",
    estimate: float | None = None,
    second_csv: Path | None = None,
) -> tuple[list[list[float]], list[float], float]:
    rows = get_rows(input_csv)
    second_rows = get_rows(second_csv) if second_csv is not None else None

    groups = _build_groups_for_source(
        rows,
        source_mode=source_mode,
        estimate=estimate,
        second_rows=second_rows,
        override_values=override_values,
    )
    if not groups:
        raise ValueError("No valid groups found to compute a repeatability metric.")

    group_metrics = [group_metric(group, method) for group in groups]
    metric = float(np.mean(group_metrics))
    return groups, group_metrics, metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute repeatability error from grouped column-10 values in a CSV."
    )
    parser.add_argument(
        "input_csv",
        help="Input CSV file path or filename in data/raw.",
    )
    parser.add_argument(
        "--method",
        choices=["range", "std"],
        default="range",
        help="Metric to compute across grouped repeated-point measurements.",
    )
    parser.add_argument(
        "--source-mode",
        choices=["ToF", "Thickness", "Speed", "Errors"],
        default="ToF",
        help="How to derive the per-row values before measuring repeatability.",
    )
    parser.add_argument(
        "--estimate",
        type=float,
        default=None,
        help="Estimate used by the Thickness or Speed source mode.",
    )
    parser.add_argument(
        "--second-csv",
        type=Path,
        default=None,
        help="Second CSV used for the Errors source mode.",
    )
    parser.add_argument(
        "--override-values-file",
        type=Path,
        default=None,
        help="Optional file containing numeric values to use instead of reading column 10.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = resolve_input_path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    override_values = None
    if args.override_values_file is not None:
        override_values = read_override_values(args.override_values_file)

    second_csv = None
    if args.second_csv is not None:
        second_csv = resolve_input_path(str(args.second_csv))

    groups, group_metrics, metric = calculate_repeatability_error(
        input_csv=input_csv,
        method=args.method,
        override_values=override_values,
        source_mode=args.source_mode,
        estimate=args.estimate,
        second_csv=second_csv,
    )

    print(f"Valid groups found: {len(groups)}")
    for idx, value in enumerate(group_metrics, start=1):
        print(f"Group {idx} {args.method} metric: {value}")
    print(f"Average {args.method} metric: {metric}")


if __name__ == "__main__":
    main()
