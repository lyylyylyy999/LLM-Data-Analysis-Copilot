import json

import pandas as pd
import pytest

from data_copilot.data_loader import (
    CSVValidationError,
    dataset_info,
    dataset_overview,
    load_csv,
)


def test_load_and_overview(tmp_path):
    path = tmp_path / "example.CSV"
    path.write_text("region,sales\nEast,10\nWest,\n", encoding="utf-8")
    df = load_csv(path)
    assert df.shape == (2, 2)
    assert df.loc[0, "sales"] == 10
    info = dataset_info(df)
    assert info == {
        "row_count": 2,
        "column_count": 2,
        "columns": ["region", "sales"],
        "dtypes": {"region": str(df.region.dtype), "sales": "float64"},
        "missing_values": {"region": 0, "sales": 1},
    }
    overview = dataset_overview(df, preview_rows=2)
    assert overview["preview"] == [
        {"region": "East", "sales": 10.0},
        {"region": "West", "sales": None},
    ]
    json.dumps(overview, allow_nan=False)
    assert dataset_overview(df, preview_rows=0)["preview"] == []


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_csv(tmp_path / "absent.csv")


def test_extension_and_directory(tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("a\n1\n")
    with pytest.raises(ValueError, match=".csv"):
        load_csv(path)
    with pytest.raises(ValueError, match="not a file"):
        load_csv(tmp_path)


@pytest.mark.parametrize(
    "content",
    [
        "",
        "\n\n",
        "a,b\n1,2,3\n",
        "a,b\n1\n",
        'a,b\n"unclosed,2',
        "a,a\n1,2",
        "a,\n1,2",
        "a,   \n1,2",
    ],
)
def test_invalid_csv(tmp_path, content):
    path = tmp_path / "bad.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(CSVValidationError):
        load_csv(path)


def test_encoding(tmp_path):
    path = tmp_path / "encoded.csv"
    path.write_text("地区,销售额\n华东,100\n", encoding="gbk")
    with pytest.raises(UnicodeDecodeError):
        load_csv(path)
    assert load_csv(path, encoding="gbk").iloc[0, 0] == "华东"


def test_bom_quoted_fields_and_header_only(tmp_path):
    path = tmp_path / "bom.csv"
    path.write_text(
        'name,n\n"a,b",2\n"line\nbreak",3\n', encoding="utf-8-sig", newline=""
    )
    df = load_csv(path)
    assert list(df.columns) == ["name", "n"]
    assert df.name.tolist() == ["a,b", "line\nbreak"]
    path.write_text("a,b\n", encoding="utf-8")
    assert load_csv(path).shape == (0, 2)


@pytest.mark.parametrize(
    "rows,error", [(-1, ValueError), (True, TypeError), (1.5, TypeError)]
)
def test_invalid_preview(rows, error):
    with pytest.raises(error):
        dataset_overview(pd.DataFrame({"x": [1]}), preview_rows=rows)


def test_preview_json_scalars():
    df = pd.DataFrame(
        {
            "integer": pd.Series([1, None], dtype="Int64"),
            "float": [float("inf"), float("nan")],
            "date": [pd.Timestamp("2025-01-01"), pd.NaT],
            "delta": [pd.Timedelta(days=1), pd.NaT],
        }
    )
    result = dataset_overview(df)
    assert result["preview"][0]["date"] == "2025-01-01T00:00:00"
    assert result["preview"][0]["integer"] == 1
    assert result["preview"][0]["float"] is None
    assert result["preview"][1]["date"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("columns", [["x", "x"], [""], [1]])
def test_invalid_dataframe_headers(columns):
    with pytest.raises(ValueError):
        dataset_info(pd.DataFrame([[1] * len(columns)], columns=columns))


@pytest.mark.parametrize("content", ["a\x00b,c\n1,2", "a,b\n1,2\x00ignored"])
def test_reject_null_bytes(tmp_path, content):
    path = tmp_path / "nul.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(CSVValidationError, match="NUL"):
        load_csv(path)
