"""Utilities for grouping repeated-point scan rows.

Grouping rule:
- Use rows with numeric TOF values in column 10 (1-based).
- Group rows by their (column 8, column 9) (1-based) value combination, regardless of
  whether that combination appears in consecutive rows.
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
    """Group rows by their (col8, col9) value combination among numeric-TOF rows."""
    data_rows = rows[1:] if skip_header else rows

    groups_by_key: dict[tuple[str, ...], list[list[str]]] = {}
    key_order: list[tuple[str, ...]] = []

    for row in data_rows:
        if parse_numeric_tof(row, tof_col=tof_col) is None:
            continue

        current_key = row_group_key(row, key_cols=key_cols)
        if current_key not in groups_by_key:
            groups_by_key[current_key] = []
            key_order.append(current_key)

        groups_by_key[current_key].append(row)

    return [groups_by_key[key] for key in key_order]


def group_row_indices_by_repeat_point(
    rows: list[list[str]],
    tof_col: int = 10,
    key_cols: Iterable[int] = (8, 9),
    skip_header: bool = True,
) -> list[list[int]]:
    """Group indices into the numeric-TOF row sequence (original order) by (col8, col9).

    Index i in the returned groups refers to the i-th row (in original file order) that has
    a numeric TOF value - i.e. the same order external override-value arrays are built in.
    """
    data_rows = rows[1:] if skip_header else rows

    indices_by_key: dict[tuple[str, ...], list[int]] = {}
    key_order: list[tuple[str, ...]] = []
    numeric_row_index = 0

    for row in data_rows:
        if parse_numeric_tof(row, tof_col=tof_col) is None:
            continue

        current_key = row_group_key(row, key_cols=key_cols)
        if current_key not in indices_by_key:
            indices_by_key[current_key] = []
            key_order.append(current_key)

        indices_by_key[current_key].append(numeric_row_index)
        numeric_row_index += 1

    return [indices_by_key[key] for key in key_order]


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
