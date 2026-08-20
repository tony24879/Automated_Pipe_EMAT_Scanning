"""Utilities for grouping repeated-point scan rows.

Grouping rule:
- Use rows with numeric TOF values in column 10 (1-based).
- Start a new group whenever column 8 or 9 (1-based) changes vs previous usable row.
"""

from __future__ import annotations

from typing import Iterable


def _to_zero_based(index_1_based: int) -> int:
    if index_1_based < 1:
        raise ValueError("Column indices must be 1-based and >= 1.")
    return index_1_based - 1


def parse_numeric_tof(row: list[str], tof_col: int = 10) -> float | None:
    """Parse TOF from one row, returning None for missing/non-numeric values."""
    tof_idx = _to_zero_based(tof_col)
    if len(row) <= tof_idx:
        return None
    text = row[tof_idx].strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def row_group_key(row: list[str], key_cols: Iterable[int] = (8, 9)) -> tuple[str, ...]:
    """Return the grouping key from one row (column values as stripped strings)."""
    key_indices = [_to_zero_based(col) for col in key_cols]
    key_values: list[str] = []
    for idx in key_indices:
        key_values.append(row[idx].strip() if len(row) > idx else "")
    return tuple(key_values)


def group_rows_by_repeat_point(
    rows: list[list[str]],
    tof_col: int = 10,
    key_cols: Iterable[int] = (8, 9),
    skip_header: bool = True,
) -> list[list[list[str]]]:
    """Group rows by consecutive runs of equal (col8, col9) among numeric-TOF rows."""
    data_rows = rows[1:] if skip_header else rows

    groups: list[list[list[str]]] = []
    current_group: list[list[str]] = []
    previous_key: tuple[str, ...] | None = None

    for row in data_rows:
        if parse_numeric_tof(row, tof_col=tof_col) is None:
            continue

        current_key = row_group_key(row, key_cols=key_cols)
        if previous_key is not None and current_key != previous_key:
            if current_group:
                groups.append(current_group)
            current_group = []

        current_group.append(row)
        previous_key = current_key

    if current_group:
        groups.append(current_group)

    return groups


def grouped_tof_values(
    grouped_rows: list[list[list[str]]],
    tof_col: int = 10,
) -> list[list[float]]:
    """Convert grouped rows into grouped TOF values from column 10."""
    values: list[list[float]] = []
    for group in grouped_rows:
        group_values: list[float] = []
        for row in group:
            parsed = parse_numeric_tof(row, tof_col=tof_col)
            if parsed is None:
                continue
            group_values.append(parsed)
        if group_values:
            values.append(group_values)
    return values
