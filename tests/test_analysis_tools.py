import json
import math

import numpy as np
import pandas as pd
import pytest

from data_copilot.analysis_tools import (
    correlation,
    dataset_info,
    describe_numeric,
    group_aggregate,
)


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "region": ["A", "A", "B", None, "C"],
            "sales": [1.0, 3.0, 5.0, None, None],
            "ads": [2.0, 6.0, 10.0, 12.0, None],
        }
    )


def test_describe_hand_computed(frame):
    result = describe_numeric(frame, "sales")
    assert result["ddof"] == 1
    assert result["statistics"]["sales"] == {
        "count": 3,
        "mean": 3.0,
        "std": 2.0,
        "min": 1.0,
        "25%": 2.0,
        "median": 3.0,
        "75%": 4.0,
        "max": 5.0,
        "unavailable_statistics": [],
    }
    assert set(describe_numeric(frame, ["sales", "ads"])["statistics"]) == {
        "sales",
        "ads",
    }


def test_undefined_describe():
    df = pd.DataFrame({"empty": [np.nan], "one": [2.0]})
    stats = describe_numeric(df, ["empty", "one"])["statistics"]
    assert stats["empty"]["count"] == 0
    assert stats["empty"]["mean"] is None
    assert stats["empty"]["median"] is None
    assert stats["one"]["std"] is None
    assert stats["one"]["unavailable_statistics"] == ["std"]
    json.dumps(stats, allow_nan=False)


@pytest.mark.parametrize("column", ["missing", "region"])
def test_column_validation(frame, column):
    for call in [
        lambda: describe_numeric(frame, column),
        lambda: group_aggregate(frame, "region", column),
        lambda: correlation(frame, column, "ads"),
        lambda: correlation(frame, "sales", column),
    ]:
        with pytest.raises(ValueError, match="Column"):
            call()


@pytest.mark.parametrize("columns", [[], ["sales", "sales"]])
def test_invalid_selection(frame, columns):
    with pytest.raises(ValueError):
        describe_numeric(frame, columns)


@pytest.mark.parametrize(
    "aggregation,expected",
    [
        ("mean", [2.0, 5.0, None, None]),
        ("sum", [4.0, 5.0, None, None]),
        ("count", [2, 1, 0, 0]),
        ("min", [1.0, 5.0, None, None]),
        ("max", [3.0, 5.0, None, None]),
    ],
)
def test_group_aggregate(frame, aggregation, expected):
    result = group_aggregate(frame, "region", "sales", aggregation)
    assert result["aggregation"] == aggregation
    assert [row["group"] for row in result["groups"]] == ["A", "B", None, "C"]
    assert [row["value"] for row in result["groups"]] == expected
    assert result["groups"][2]["status"] == (
        "ok" if aggregation == "count" else "undefined_or_nonfinite"
    )
    json.dumps(result, allow_nan=False)


def test_categorical_observed():
    df = pd.DataFrame(
        {
            "key": pd.Categorical(["A", None], categories=["A", "unused"]),
            "x": [2, 4],
        }
    )
    assert group_aggregate(df, "key", "x", "sum")["groups"] == [
        {"group": "A", "value": 2, "status": "ok"},
        {"group": None, "value": 4, "status": "ok"},
    ]


def test_invalid_group_or_method(frame):
    with pytest.raises(ValueError, match="does not exist"):
        group_aggregate(frame, "absent", "sales")
    with pytest.raises(ValueError, match="Unsupported aggregation"):
        group_aggregate(frame, "region", "sales", "median")
    with pytest.raises(ValueError, match="Unsupported correlation"):
        correlation(frame, "sales", "ads", "spearman")


def test_correlation_pairwise(frame):
    result = correlation(frame, "sales", "ads")
    assert result["coefficient"] == pytest.approx(1.0)
    assert result["sample_size"] == 3
    assert result["status"] == "ok"
    df = pd.DataFrame({"x": [1, 2, 3], "y": [3, 2, 1]})
    assert correlation(df, "x", "y")["coefficient"] == pytest.approx(-1.0)
    assert correlation(df, "x", "x")["coefficient"] == pytest.approx(1.0)
    df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 3, 2]})
    assert correlation(df, "x", "y")["coefficient"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "x,y,n,status",
    [
        ([1.0, np.nan], [2.0, 3.0], 1, "insufficient_samples"),
        ([np.nan], [2.0], 0, "insufficient_samples"),
        ([1, 1, 1], [2, 3, 4], 3, "zero_variance"),
        ([1, 2, 3], [2, 2, 2], 3, "zero_variance"),
    ],
)
def test_undefined_correlation(x, y, n, status):
    result = correlation(pd.DataFrame({"x": x, "y": y}), "x", "y")
    assert result["coefficient"] is None
    assert result["sample_size"] == n
    assert result["status"] == status
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "values", [[True, False], [1j, 2j], ["1", "2"], [1.0, math.inf]]
)
def test_reject_invalid_numeric_values(values):
    df = pd.DataFrame({"x": values, "key": ["a", "b"]})
    for call in [
        lambda: describe_numeric(df, "x"),
        lambda: group_aggregate(df, "key", "x"),
        lambda: correlation(df, "x", "x"),
    ]:
        with pytest.raises(ValueError):
            call()


def test_json_nullable_and_no_mutation():
    df = pd.DataFrame(
        {
            "key": [pd.Timestamp("2025-01-01"), pd.NaT, pd.NaT],
            "x": pd.Series([1, None, 3], dtype="Int64"),
            "y": pd.Series([2, None, 6], dtype="Float64"),
        }
    )
    original = df.copy(deep=True)
    for result in [
        dataset_info(df),
        describe_numeric(df, "x"),
        group_aggregate(df, "key", "x"),
        correlation(df, "x", "y"),
    ]:
        json.dumps(result, allow_nan=False)
    pd.testing.assert_frame_equal(df, original)


def test_empty_dataframe():
    df = pd.DataFrame({"key": pd.Series(dtype="str"), "x": pd.Series(dtype="float64")})
    assert dataset_info(df)["row_count"] == 0
    assert group_aggregate(df, "key", "x")["groups"] == []
    assert describe_numeric(df, "x")["statistics"]["x"]["count"] == 0
    assert correlation(df, "x", "x")["status"] == "insufficient_samples"


@pytest.mark.parametrize(
    "columns", [None, 1, b"sales", {"sales": True}, [1], [["sales"]]]
)
def test_invalid_column_selection_types(frame, columns):
    with pytest.raises(TypeError, match="column"):
        describe_numeric(frame, columns)


@pytest.mark.parametrize(
    "column,error", [(None, TypeError), (["sales"], TypeError), (" ", ValueError)]
)
def test_invalid_column_names(frame, column, error):
    for call in [
        lambda: group_aggregate(frame, column, "sales"),
        lambda: group_aggregate(frame, "region", column),
        lambda: correlation(frame, column, "ads"),
        lambda: correlation(frame, "sales", column),
    ]:
        with pytest.raises(error, match="Column name"):
            call()


@pytest.mark.parametrize("operation", [None, [], 1])
def test_invalid_operation_types(frame, operation):
    with pytest.raises(TypeError, match="aggregation"):
        group_aggregate(frame, "region", "sales", operation)
    with pytest.raises(TypeError, match="method"):
        correlation(frame, "sales", "ads", operation)


def test_repository_sales_sample():
    from pathlib import Path

    from data_copilot.data_loader import load_csv

    path = Path(__file__).resolve().parents[1] / "data/samples/sales.csv"
    df = load_csv(path)
    assert df.shape == (40, 6)
    assert dataset_info(df)["missing_values"] == {
        "order_id": 0,
        "region": 1,
        "category": 0,
        "sales": 2,
        "advertising_spend": 2,
        "quantity": 0,
    }
    description = describe_numeric(df, "sales")
    assert description["statistics"]["sales"]["mean"] == pytest.approx(
        694.6973684210526
    )
    groups = group_aggregate(df, "region", "sales")
    assert groups["groups"][0] == {"group": "East", "value": 680.375, "status": "ok"}
    result = correlation(df, "sales", "advertising_spend")
    assert result["sample_size"] == 36
    assert result["coefficient"] == pytest.approx(0.9409742630262548)
    for output in [dataset_info(df), description, groups, result]:
        json.dumps(output, allow_nan=False)
