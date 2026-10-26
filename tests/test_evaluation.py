"""Offline verification of the evaluator using real statistical tools only."""

import copy
import importlib.util
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from data_copilot.analysis_tools import (
    correlation,
    dataset_info,
    describe_numeric,
    group_aggregate,
)


@pytest.fixture
def evaluator(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("Evaluation tests must remain offline"))
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr("dotenv.load_dotenv", forbidden)
    monkeypatch.setattr("openai.OpenAI", forbidden)
    service = Mock(side_effect=AssertionError("Service must be explicitly mocked"))
    monkeypatch.setattr("data_copilot.analysis_service.analyze_question", service)
    path = Path(__file__).resolve().parents[1] / "examples/run_evaluation.py"
    spec = importlib.util.spec_from_file_location("evaluation_offline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(module=module, service=service, forbidden=forbidden)


def test_import_does_not_request_or_write(evaluator):
    evaluator.service.assert_not_called()
    evaluator.forbidden.assert_not_called()


def requirement(*, name="describe_numeric", arguments=None, checks=None):
    return {
        "name": name,
        "arguments": {"columns": ["sales"]} if arguments is None else arguments,
        "checks": {"statistics.sales.count": 38} if checks is None else checks,
    }


def tool(*, name="describe_numeric", arguments=None, result=None, identifier="call_1"):
    return {
        "name": name,
        "tool_call_id": identifier,
        "arguments": {"columns": ["sales"]} if arguments is None else arguments,
        "result": {"statistics": {"sales": {"count": 38}}}
        if result is None
        else result,
    }


@pytest.mark.parametrize(
    "actual,expected,passed",
    [
        (694.69736842, {"value": 694.697368421, "abs_tol": 1e-6}, True),
        (694.70, {"value": 694.697368421, "abs_tol": 1e-6}, False),
        (100.01, {"value": 100.0, "rel_tol": 0.001}, True),
        (None, None, True),
        (0, None, False),
        (None, 0, False),
        (True, 1, False),
        (38.0, 38, False),
        ("38", 38, False),
        ("zero_variance", "zero_variance", True),
        ("ok", "zero_variance", False),
        (float("nan"), {"value": 1.0}, False),
    ],
)
def test_numeric_and_state_comparison(evaluator, actual, expected, passed):
    assert evaluator.module._compare(actual, expected) is passed


@pytest.mark.parametrize(
    "actual",
    [
        [],
        [tool(name="dataset_info", arguments={})],
        [tool(arguments={"columns": ["quantity"]})],
    ],
)
def test_tool_name_alone_is_not_enough(evaluator, actual):
    result = evaluator.module.check_tools(actual, [requirement()])
    assert result["passed"] is False
    assert result["requirements"][0]["tool_call_id"] is None


def test_extra_calls_and_columns_and_order_are_allowed(evaluator):
    calls = [
        tool(name="dataset_info", arguments={}, identifier="extra"),
        tool(arguments={"columns": ["quantity", "sales"]}, identifier="needed"),
    ]
    result = evaluator.module.check_tools(calls, [requirement()])
    assert result["passed"] is True
    assert result["requirements"][0]["tool_call_id"] == "needed"


def test_repeated_tool_uses_matching_successful_call(evaluator):
    calls = [
        tool(result={"statistics": {"sales": {"count": 0}}}),
        tool(identifier="correct"),
    ]
    result = evaluator.module.check_tools(calls, [requirement()])
    assert result["passed"] is True
    assert result["requirements"][0]["tool_call_id"] == "correct"


def test_required_checks_cannot_be_combined_from_different_calls(evaluator):
    expected = requirement(checks={"count": 38, "mean": {"value": 20.0}})
    calls = [
        tool(result={"count": 38, "mean": 0}),
        tool(result={"count": 0, "mean": 20.0}),
    ]
    assert evaluator.module.check_tools(calls, [expected])["passed"] is False


@pytest.mark.parametrize(
    "result,passed",
    [({}, False), ({"coefficient": None}, True), ({"coefficient": 0}, False)],
)
def test_missing_field_is_not_the_same_as_null(evaluator, result, passed):
    expected = requirement(checks={"coefficient": None})
    actual = evaluator.module.check_tools([tool(result=result)], [expected])
    assert actual["passed"] is passed
    assert actual["requirements"][0]["checks"][0]["found"] is ("coefficient" in result)


def test_correlation_parameters_can_be_reversed(evaluator):
    expected = requirement(
        name="correlation",
        arguments={"x_column": "sales", "y_column": "ads", "method": "pearson"},
        checks={"sample_size": 36},
    )
    call = tool(
        name="correlation",
        arguments={"x_column": "ads", "y_column": "sales", "method": "pearson"},
        result={"sample_size": 36},
    )
    assert evaluator.module.check_tools([call], [expected])["passed"] is True
    call["arguments"]["y_column"] = "quantity"
    assert evaluator.module.check_tools([call], [expected])["passed"] is False


def test_no_expected_tools_cannot_pass(evaluator):
    assert evaluator.module.check_tools([], [])["passed"] is False


def real_tool_service(frame, question):
    if "West 与 North" in question or "region" in question or "category" in question:
        group = "category" if "category" in question else "region"
        aggregations = (
            ["sum", "count"]
            if "West 与 North" in question
            else ["sum" if group == "category" else "mean"]
        )
        calls = []
        for aggregation in aggregations:
            args = {
                "group_column": group,
                "target_column": "sales",
                "aggregation": aggregation,
            }
            calls.append(
                tool(
                    name="group_aggregate",
                    arguments=args,
                    result=group_aggregate(frame, **args),
                    identifier=aggregation,
                )
            )
        return {
            "answer": "(!256)",
            "model": "actual-offline-model",
            "finish_reason": "stop",
            "tool_results": calls,
        }
    if "行" in question:
        name, args, result = "dataset_info", {}, dataset_info(frame)
    elif "标准差" in question:
        name, args = "describe_numeric", {"columns": ["sales"]}
        result = describe_numeric(frame, ["sales"])
    else:
        x, y = ("x", "y") if "x" in frame.columns else ("sales", "advertising_spend")
        name, args = "correlation", {"x_column": x, "y_column": y, "method": "pearson"}
        result = correlation(frame, x, y)
    return {
        "answer": "(!256)",
        "model": "actual-offline-model",
        "finish_reason": "stop",
        "tool_results": [tool(name=name, arguments=args, result=result)],
    }


def test_repository_cases_are_independently_checkable_with_real_tools(evaluator):
    evaluator.service.side_effect = real_tool_service
    report = evaluator.module.run_evaluation()
    assert report["summary"] == {
        "total": 7,
        "passed": 7,
        "failed": 0,
        "human_review_pending": 7,
    }
    assert evaluator.service.call_count == 7
    assert {case["id"] for case in report["cases"]} == {
        "dataset_size",
        "sales_descriptive",
        "pearson_correlation",
        "zero_variance",
        "region_mean",
        "category_sum",
        "group_missing_sum_count",
    }
    for record in report["cases"]:
        assert record["answer"] == "(!256)"
        assert record["model"] == "actual-offline-model"
        assert record["human_review"]["status"] == "pending"
        assert record["human_review"]["criteria"]
        assert record["error_type"] is None
        assert all(item["checks"] for item in record["automatic_check"]["requirements"])
    json.dumps(report, allow_nan=False)


def test_one_case_failure_continues_without_leaking_exception(evaluator):
    count = 0

    def service(frame, question):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("fake-secret-key private-server-path raw-upstream-body")
        return real_tool_service(frame, question)

    evaluator.service.side_effect = service
    report = evaluator.module.run_evaluation()
    assert report["summary"] == {
        "total": 7,
        "passed": 6,
        "failed": 1,
        "human_review_pending": 7,
    }
    failed = report["cases"][1]
    assert failed["error_type"] == "RuntimeError"
    assert failed["automatic_check"]["passed"] is False
    text = json.dumps(report, allow_nan=False)
    for value in ("fake-secret-key", "private-server-path", "raw-upstream-body"):
        assert value not in text
    assert evaluator.service.call_count == 7


def test_paths_are_independent_of_cwd_and_reject_escape(
    evaluator, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    evaluator.service.side_effect = real_tool_service
    assert evaluator.module.run_evaluation()["summary"]["passed"] == 7
    with pytest.raises(ValueError, match="项目目录"):
        evaluator.module._csv_path("../outside.csv")


def test_report_writer_serializes_and_prints_summary(
    evaluator, monkeypatch, tmp_path, capsys
):
    evaluator.service.side_effect = real_tool_service
    target = tmp_path / "reports" / "report.json"
    monkeypatch.setattr(sys, "argv", ["run_evaluation", "--output", str(target)])
    evaluator.module.main()
    report = json.loads(target.read_text(encoding="utf-8"))
    assert report["summary"]["passed"] == 7
    assert report["summary"]["human_review_pending"] == 7
    assert "7 通过，0 失败" in capsys.readouterr().out
    json.dumps(report, allow_nan=False)


def test_missing_dataset_fails_before_model_request(evaluator):
    record = evaluator.module.evaluate_case(
        {
            "id": "absent",
            "csv": "evals/fixtures/does_not_exist.csv",
            "question": "问题",
            "expected_tools": [requirement()],
        }
    )
    assert record["error_type"] == "FileNotFoundError"
    assert record["automatic_check"]["passed"] is False
    evaluator.service.assert_not_called()


def grouped_requirement(aggregation="sum"):
    return {
        "name": "group_aggregate",
        "arguments": {
            "group_column": "region",
            "target_column": "sales",
            "aggregation": aggregation,
        },
        "checks": {"aggregation": aggregation},
        "expected_groups": [
            {"group": "West", "value": None if aggregation == "sum" else 0},
            {"group": "North", "value": 0 if aggregation == "sum" else 2},
            {"group": None, "value": 20 if aggregation == "sum" else 1},
        ],
    }


def grouped_call(aggregation="sum", *, identifier="group"):
    expected = grouped_requirement(aggregation)
    return tool(
        name="group_aggregate",
        arguments=expected["arguments"],
        identifier=identifier,
        result={
            "aggregation": aggregation,
            "groups": copy.deepcopy(expected["expected_groups"]),
        },
    )


def test_groups_match_by_key_and_report_values_not_list_order(evaluator):
    call = grouped_call()
    call["result"]["groups"].reverse()
    check = evaluator.module.check_tools([call], [grouped_requirement()])
    assert check["passed"] is True
    groups = check["requirements"][0]["group_checks"]
    assert groups == [
        {
            "group": key,
            "expected": value,
            "actual": value,
            "match_count": 1,
            "value_found": True,
            "passed": True,
        }
        for key, value in [("West", None), ("North", 0), (None, 20)]
    ]
    json.dumps(check, allow_nan=False)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_group",
        "duplicate_group",
        "duplicate_null",
        "string_null",
        "missing_value",
        "null_to_zero",
        "zero_to_null",
        "bool_zero",
    ],
)
def test_invalid_group_result_cannot_pass(evaluator, mutation):
    call = grouped_call()
    rows = call["result"]["groups"]
    if mutation == "missing_group":
        rows.pop(0)
    elif mutation == "duplicate_group":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "duplicate_null":
        rows.append(copy.deepcopy(rows[2]))
    elif mutation == "string_null":
        rows[2]["group"] = "null"
    elif mutation == "missing_value":
        del rows[0]["value"]
    elif mutation == "null_to_zero":
        rows[0]["value"] = 0
    elif mutation == "zero_to_null":
        rows[1]["value"] = None
    else:
        rows[1]["value"] = False
    report = evaluator.module.check_tools([call], [grouped_requirement()])
    assert report["passed"] is False
    assert any(
        not check["passed"] for check in report["requirements"][0]["group_checks"]
    )


def test_duplicate_group_report_retains_all_candidate_values(evaluator):
    call = grouped_call()
    call["result"]["groups"].append({"group": "West", "value": 0})
    report = evaluator.module.check_tools([call], [grouped_requirement()])
    west = report["requirements"][0]["group_checks"][0]
    assert west["match_count"] == 2
    assert west["actual"] == [None, 0]
    assert west["passed"] is False


@pytest.mark.parametrize("rows", [None, {}, "groups", []])
def test_missing_or_non_list_groups_fail(evaluator, rows):
    call = grouped_call()
    call["result"]["groups"] = rows
    result = evaluator.module.check_tools([call], [grouped_requirement()])
    assert result["passed"] is False
    assert all(
        item["match_count"] == 0 for item in result["requirements"][0]["group_checks"]
    )


def test_correct_groups_do_not_override_failed_normal_check(evaluator):
    call = grouped_call()
    call["result"]["aggregation"] = "count"
    result = evaluator.module.check_tools([call], [grouped_requirement()])
    assert result["passed"] is False
    requirement_result = result["requirements"][0]
    assert all(item["passed"] for item in requirement_result["group_checks"])
    assert requirement_result["checks"][0]["passed"] is False


def test_normal_checks_and_groups_must_pass_in_same_call(evaluator):
    wrong_groups, wrong_normal = (
        grouped_call(identifier="a"),
        grouped_call(identifier="b"),
    )
    wrong_groups["result"]["groups"] = []
    wrong_normal["result"]["aggregation"] = "count"
    assert (
        evaluator.module.check_tools(
            [wrong_groups, wrong_normal], [grouped_requirement()]
        )["passed"]
        is False
    )


def test_sum_and_count_are_matched_independently_even_when_reordered(evaluator):
    sum_call, count_call = (
        grouped_call(identifier="sum"),
        grouped_call("count", identifier="count"),
    )
    expected = [grouped_requirement(), grouped_requirement("count")]
    result = evaluator.module.check_tools([count_call, sum_call], expected)
    assert result["passed"] is True
    assert [item["tool_call_id"] for item in result["requirements"]] == ["sum", "count"]
    assert evaluator.module.check_tools([sum_call], expected)["passed"] is False
    # Mislabeling count values as sum cannot satisfy the sum expectation.
    sum_call["result"]["groups"] = count_call["result"]["groups"]
    assert (
        evaluator.module.check_tools([count_call, sum_call], expected)["passed"]
        is False
    )


@pytest.mark.parametrize("delta,passed", [(0.0000001, True), (0.01, False)])
def test_group_float_tolerance(evaluator, delta, passed):
    expected = grouped_requirement()
    expected["expected_groups"] = [
        {"group": "South", "value": {"value": 699.333333333, "abs_tol": 1e-6}}
    ]
    call = grouped_call()
    call["result"]["groups"] = [{"group": "South", "value": 699.333333333 + delta}]
    assert evaluator.module.check_tools([call], [expected])["passed"] is passed


def test_group_only_expectation_works_without_normal_checks(evaluator):
    expected = grouped_requirement()
    del expected["checks"]
    assert evaluator.module.check_tools([grouped_call()], [expected])["passed"] is True


def test_equivalent_float_sum_zero_is_allowed_but_not_null(evaluator):
    call = grouped_call()
    call["result"]["groups"][1]["value"] = 0.0
    assert (
        evaluator.module.check_tools([call], [grouped_requirement()])["passed"] is True
    )


def test_group_fixture_matches_hand_calculated_sums_and_counts(evaluator):
    from data_copilot.data_loader import load_csv

    frame = load_csv(evaluator.module.PROJECT_ROOT / "evals/fixtures/group_missing.csv")
    expected = {
        "sum": [("East", 10), ("West", None), ("North", 0), (None, 20)],
        "count": [("East", 1), ("West", 0), ("North", 2), (None, 1)],
    }
    for aggregation, values in expected.items():
        result = group_aggregate(frame, "region", "sales", aggregation)
        assert [(row["group"], row["value"]) for row in result["groups"]] == values
