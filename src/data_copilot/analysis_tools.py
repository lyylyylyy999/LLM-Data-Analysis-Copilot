"""参数显式传入、不修改输入数据且返回标准 JSON 兼容结果的统计工具。"""

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
    """对选定的实数数值列计算描述统计，跳过缺失值，标准差使用 ddof=1。

    无法计算或溢出的统计量返回 null，并将名称列入 unavailable_statistics。
    不自动将字符串转换为数字。
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
    """对实数数值列分组聚合，保留缺失分组键为 null，按首次出现顺序返回。

    count 只统计目标列的非缺失值。sum 使用 min_count=1，使全缺失组返回 null，
    避免被误解为零。仅返回实际出现的分组，分类类型的分组列也遵循此规则。
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
    """计算 Pearson 相关系数，仅使用两列均非缺失的样本；相关关系不代表因果。

    有效样本对少于两对，或任一变量为常量时，返回 null 和明确状态。
    输入包含无穷值时直接报错，不静默删除这些值。
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
        coefficient = _json_value(x.corr(y, method="pearson"))
        status = "ok" if coefficient is not None else "nonfinite_result"
    return {
        "x_column": x_column,
        "y_column": y_column,
        "method": method,
        "coefficient": coefficient,
        "sample_size": int(n),
        "status": status,
    }
