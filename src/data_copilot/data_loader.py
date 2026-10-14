"""严格校验并读取 CSV，生成兼容 JSON 的数据集概览。"""

import csv
import math
from collections import Counter
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class CSVValidationError(ValueError):
    """输入文件为空、格式错误或表头无效时抛出的异常。"""


def _json_value(value: Any) -> Any:
    """将标量或容器中的值转换为 JSON 兼容类型，不对有限数值舍入。"""
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}")


def _validate_frame(dataframe: pd.DataFrame) -> None:
    """要求列名为唯一且非空白的字符串，确保 JSON 字段和工具参数稳定。"""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    names = list(dataframe.columns)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Column names must be nonblank strings")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate column names are not supported")


def load_csv(file_path: str | Path, *, encoding: str = "utf-8-sig") -> pd.DataFrame:
    """读取逗号分隔的 CSV，拒绝无效表头及字段数量不一致的记录。

    编码错误和操作系统错误直接向调用方抛出，不自动检测编码或清洗数据。
    由 Pandas 推断数据类型，并使用其默认的缺失值标记。
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"CSV path is not a file: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file: {path}")
    try:
        with path.open("r", encoding=encoding, newline="") as source:
            reader = csv.reader(source, strict=True)
            header = next((row for row in reader if row), None)
            if header is None:
                raise CSVValidationError("CSV file is empty")
            if any("\x00" in name for name in header):
                raise CSVValidationError("CSV contains a NUL byte in its header")
            if any(not name.strip() for name in header):
                raise CSVValidationError("CSV contains an empty column name")
            duplicates = [name for name, n in Counter(header).items() if n > 1]
            if duplicates:
                raise CSVValidationError(f"Duplicate CSV column names: {duplicates}")
            for row in reader:
                if any("\x00" in field for field in row):
                    raise CSVValidationError("CSV contains a NUL byte in its data")
                if row and len(row) != len(header):
                    raise CSVValidationError(
                        f"CSV record ending at line {reader.line_num} has "
                        f"{len(row)} fields; expected {len(header)}"
                    )
        return pd.read_csv(path, encoding=encoding, skip_blank_lines=True)
    except (csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise CSVValidationError(f"Cannot parse CSV {path}: {exc}") from exc


def dataset_info(dataframe: pd.DataFrame) -> dict[str, Any]:
    """返回兼容 JSON 的行列数、列名、数据类型和各列缺失值数量。"""
    _validate_frame(dataframe)
    return {
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "columns": list(dataframe.columns),
        "dtypes": {name: str(dtype) for name, dtype in dataframe.dtypes.items()},
        "missing_values": {
            name: int(count) for name, count in dataframe.isna().sum().items()
        },
    }


def dataset_overview(
    dataframe: pd.DataFrame, *, preview_rows: int = 5
) -> dict[str, Any]:
    """添加按行组织的预览；时间戳转为 ISO 字符串，非有限数值转为 null。"""
    if isinstance(preview_rows, bool) or not isinstance(preview_rows, int):
        raise TypeError("preview_rows must be an integer")
    if preview_rows < 0:
        raise ValueError("preview_rows must be nonnegative")
    return {
        **dataset_info(dataframe),
        "preview": _json_value(dataframe.head(preview_rows).to_dict(orient="records")),
    }
