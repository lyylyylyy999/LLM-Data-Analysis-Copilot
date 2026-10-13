"""Explicit, non-mutating statistical tools with standard JSON results."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from data_copilot.data_loader import _json_value, _validate_frame, dataset_info

__all__ = ["dataset_info", "describe_numeric", "group_aggregate", "correlation"]


def _require_column(dataframe: pd.DataFrame, column: str) -> None:
    _validate_frame(dataframe)
    if not isinstance(column, str):
        raise TypeError("Column name must be a string")
    if not column.strip():
        raise ValueError("Column name must not be blank")
    if column not in dataframe.columns:
        raise ValueError(f"Column does not exist: {column!r}")


def _require_numeric(dataframe: pd.DataFrame, column: str) -> None:
    _require_column(dataframe, column)
    series = dataframe[column]
    if (
        not is_numeric_dtype(series.dtype)
        or is_bool_dtype(series.dtype)
        or is_complex_dtype(series.dtype)
    ):
        raise ValueError(f"Column must contain real numeric values: {column!r}")
    if not np.isfinite(series.dropna()).all():
        raise ValueError(f"Column contains infinite values: {column!r}")


def describe_numeric(
    dataframe: pd.DataFrame, columns: str | Sequence[str]
) -> dict[str, Any]:
    """Describe selected real numeric columns; skip missing values, std ddof=1.

    Undefined or overflowed statistics become null, with their names listed
    in unavailable_statistics. No string-to-number coercion is performed.
    """
    if not isinstance(columns, (str, Sequence)) or isinstance(columns, bytes):
        raise TypeError("columns must be a column name or a sequence of names")
    names = [columns] if isinstance(columns, str) else list(columns)
    if any(not isinstance(name, str) for name in names):
        raise TypeError("Each column name must be a string")
    if not names or len(set(names)) != len(names):
        raise ValueError("Select at least one column, without duplicates")
    result = {}
    for name in names:
        _require_numeric(dataframe, name)
        stats = dataframe[name].describe().to_dict()
        stats["median"] = stats.pop("50%")
        stats["count"] = int(stats["count"])
        safe_stats = _json_value(stats)
        result[name] = {
            **safe_stats,
            "unavailable_statistics": [k for k, v in safe_stats.items() if v is None],
        }
    return {"ddof": 1, "statistics": result}


def group_aggregate(
    dataframe: pd.DataFrame,
    group_column: str,
    target_column: str,
    aggregation: str = "mean",
) -> dict[str, Any]:
    """Aggregate real numeric values; keep missing keys as null, in first-seen order.

    count counts nonmissing targets. sum uses min_count=1, so all-missing
    groups return null rather than a misleading zero. Only observed groups
    are returned, including for categorical columns.
    """
    if not isinstance(aggregation, str):
        raise TypeError("aggregation must be a string")
    if aggregation not in {"mean", "sum", "count", "min", "max"}:
        raise ValueError(f"Unsupported aggregation: {aggregation!r}")
    _require_column(dataframe, group_column)
    _require_numeric(dataframe, target_column)
    grouped = dataframe.groupby(group_column, dropna=False, sort=False, observed=True)[
        target_column
    ]
    values = (
        grouped.sum(min_count=1) if aggregation == "sum" else grouped.agg(aggregation)
    )
    rows = []
    for key, value in values.items():
        safe_value = _json_value(value)
        rows.append(
            {
                "group": _json_value(key),
                "value": safe_value,
                "status": "ok" if safe_value is not None else "undefined_or_nonfinite",
            }
        )
    return {
        "group_column": group_column,
        "target_column": target_column,
        "aggregation": aggregation,
        "groups": rows,
    }


def correlation(
    dataframe: pd.DataFrame,
    x_column: str,
    y_column: str,
    method: str = "pearson",
) -> dict[str, Any]:
    """Pearson correlation with pairwise missing removal; never implies causation.

    Fewer than two complete pairs or a constant variable yields null with
    an explicit status. Infinite inputs are rejected, not silently dropped.
    """
    if not isinstance(method, str):
        raise TypeError("method must be a string")
    if method != "pearson":
        raise ValueError(f"Unsupported correlation method: {method!r}")
    _require_numeric(dataframe, x_column)
    _require_numeric(dataframe, y_column)
    pairs = dataframe[[x_column, y_column]].dropna()
    n = len(pairs)
    x, y = pairs.iloc[:, 0], pairs.iloc[:, 1]
    coefficient = None
    if n < 2:
        status = "insufficient_samples"
    elif x.nunique() < 2 or y.nunique() < 2:
        status = "zero_variance"
    else:
        coefficient = _json_value(x.corr(y, method=method))
        status = "ok" if coefficient is not None else "nonfinite_result"
    return {
        "x_column": x_column,
        "y_column": y_column,
        "method": method,
        "coefficient": coefficient,
        "sample_size": int(n),
        "status": status,
    }
