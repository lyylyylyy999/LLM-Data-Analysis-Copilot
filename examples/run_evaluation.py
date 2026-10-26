"""运行少量人工制定的模型评估案例，保留回答供人工审查。"""

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from data_copilot.analysis_service import analyze_question
from data_copilot.data_loader import load_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = PROJECT_ROOT / "evals" / "cases.json"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "evals" / "reports"


def _csv_path(relative_path: str) -> Path:
    """评估 CSV 只能引用项目内路径；不依赖运行时工作目录。"""
    path = (PROJECT_ROOT / relative_path).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError("评估 CSV 必须位于项目目录内")
    return path


def _arguments_match(actual: Any, expected: dict[str, Any]) -> bool:
    """检查语义参数；描述统计允许更多列，相关的两列允许互换。"""
    if not isinstance(actual, dict):
        return False
    if {"x_column", "y_column"}.issubset(expected):
        if actual.get("method") != expected.get("method"):
            return False
        actual_pair = (actual.get("x_column"), actual.get("y_column"))
        expected_pair = (expected["x_column"], expected["y_column"])
        if not all(isinstance(item, str) for item in (*actual_pair, *expected_pair)):
            return False
        return set(actual_pair) == set(expected_pair)

    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if key == "columns" and isinstance(expected_value, list):
            if not isinstance(actual_value, list) or not all(
                isinstance(item, str) for item in actual_value
            ):
                return False
            if not set(expected_value).issubset(actual_value):
                return False
        elif actual_value != expected_value:
            return False
    return True


def _lookup_path(result: dict[str, Any], path: str) -> tuple[bool, Any]:
    current: Any = result
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return False, None
        current = current[key]
    return True, current


def _compare(actual: Any, expected_spec: Any) -> bool:
    """整数和状态精确比较；仅浮点数接受显式容差。"""
    if isinstance(expected_spec, dict) and "value" in expected_spec:
        expected = expected_spec["value"]
        if isinstance(expected, bool) or not isinstance(expected, (int, float)):
            return False
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            return False
        return math.isclose(
            actual,
            expected,
            rel_tol=float(expected_spec.get("rel_tol", 0.0)),
            abs_tol=float(expected_spec.get("abs_tol", 1e-6)),
        )
    if expected_spec is None:
        return actual is None
    if isinstance(expected_spec, bool):
        return type(actual) is bool and actual == expected_spec
    if isinstance(expected_spec, int):
        return type(actual) is int and actual == expected_spec
    if isinstance(expected_spec, str):
        return type(actual) is str and actual == expected_spec
    return type(actual) is type(expected_spec) and actual == expected_spec


def _group_keys_equal(actual: Any, expected: Any) -> bool:
    """JSON 分组键按值和类型比较；null 不等于字符串 "null"。"""
    return type(actual) is type(expected) and actual == expected


def _compare_group_value(actual: Any, expected: Any) -> bool:
    """允许聚合的整数结果表示为 0.0，同时严格区分 null、0 和布尔值。"""
    if type(expected) is int and type(actual) in (int, float):
        return actual == expected and (type(actual) is int or math.isfinite(actual))
    return _compare(actual, expected)


def _check_expected_groups(
    result_data: Any, expected_groups: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """以分组键而非列表位置定位记录；缺失或重复的键均不合格。"""
    rows = result_data.get("groups") if isinstance(result_data, dict) else None
    if not isinstance(rows, list):
        rows = []

    group_checks: list[dict[str, Any]] = []
    for expectation in expected_groups:
        expected_group = expectation["group"]
        expected_value = expectation["value"]
        matches = [
            row
            for row in rows
            if isinstance(row, dict)
            and "group" in row
            and _group_keys_equal(row["group"], expected_group)
        ]
        match_count = len(matches)
        # 重复组同时展示所有候选值，方便从报告中定位问题。
        if match_count == 1:
            actual_value = matches[0].get("value")
            value_found = "value" in matches[0]
        elif match_count > 1:
            actual_value = [row.get("value") for row in matches]
            value_found = all("value" in row for row in matches)
        else:
            actual_value = None
            value_found = False
        group_checks.append(
            {
                "group": expected_group,
                "expected": expected_value,
                "actual": actual_value,
                "match_count": match_count,
                "value_found": value_found,
                "passed": (
                    match_count == 1
                    and value_found
                    and _compare_group_value(actual_value, expected_value)
                ),
            }
        )
    return group_checks


def _check_candidate(
    tool_result: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    result_data = tool_result.get("result")
    checks: list[dict[str, Any]] = []
    for path, expected_value in expected.get("checks", {}).items():
        if isinstance(result_data, dict):
            found, actual_value = _lookup_path(result_data, path)
        else:
            found, actual_value = False, None
        passed = found and _compare(actual_value, expected_value)
        checks.append(
            {
                "path": path,
                "expected": expected_value,
                "actual": actual_value,
                "found": found,
                "passed": passed,
            }
        )
    group_checks = _check_expected_groups(
        result_data, expected.get("expected_groups", [])
    )
    return {
        "name": expected["name"],
        "expected_arguments": expected["arguments"],
        "tool_call_id": tool_result.get("tool_call_id"),
        "passed": (
            all(item["passed"] for item in checks)
            and all(item["passed"] for item in group_checks)
        ),
        "checks": checks,
        "group_checks": group_checks,
    }


def check_tools(
    tool_results: list[dict[str, Any]], expected_tools: list[dict[str, Any]]
) -> dict[str, Any]:
    """按工具名和参数寻找可匹配调用，不依赖调用位置或次数。"""
    if not expected_tools:
        return {
            "passed": False,
            "requirements": [],
            "reason": "评估案例未定义预期工具，不能自动判定通过",
        }
    requirements: list[dict[str, Any]] = []
    for expected in expected_tools:
        candidates = [
            item
            for item in tool_results
            if item.get("name") == expected["name"]
            and _arguments_match(item.get("arguments"), expected["arguments"])
        ]
        if not candidates:
            requirements.append(
                {
                    "name": expected["name"],
                    "expected_arguments": expected["arguments"],
                    "tool_call_id": None,
                    "passed": False,
                    "checks": [],
                    "group_checks": [],
                    "reason": "未找到工具名称和参数都匹配的调用",
                }
            )
            continue
        # 同名工具可能执行多次，只要有一条满足全部预期统计值即可。
        evaluated = [_check_candidate(item, expected) for item in candidates]
        matched = next((item for item in evaluated if item["passed"]), evaluated[0])
        requirements.append(matched)
    return {
        "passed": all(item["passed"] for item in requirements),
        "requirements": requirements,
    }


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    """每个案例独立运行；异常仅记录类型，不泄露上游原始信息。"""
    case_fields = case if isinstance(case, dict) else {}
    record: dict[str, Any] = {
        "id": case_fields.get("id"),
        "model": None,
        "answer": None,
        "tool_results": [],
        "automatic_check": {"passed": False, "requirements": []},
        "human_review": {
            "status": "pending",
            "criteria": case_fields.get("human_review", []),
        },
        "error_type": None,
    }
    try:
        if not isinstance(case, dict):
            raise TypeError("评估案例必须是 JSON 对象")
        frame = load_csv(_csv_path(case["csv"]))
        service_result = analyze_question(frame, case["question"])
        record["model"] = service_result["model"]
        record["answer"] = service_result["answer"]
        record["tool_results"] = service_result["tool_results"]
        record["automatic_check"] = check_tools(
            service_result["tool_results"], case["expected_tools"]
        )
    except Exception as exc:  # noqa: BLE001 - 必须继续运行后续案例
        record["error_type"] = type(exc).__name__
    return record


def run_evaluation(cases_path: Path = DEFAULT_CASES) -> dict[str, Any]:
    """运行所有案例；自动通过仅表示工具及数值检查通过。"""
    cases_config = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = cases_config["cases"]
    results = [evaluate_case(case) for case in cases]
    passed = sum(item["automatic_check"]["passed"] for item in results)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "human_review_pending": len(results),
        },
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="运行模型工具选择与数值评估")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = run_evaluation(args.cases)
    output = args.output
    if output is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        output = DEFAULT_REPORT_DIR / f"evaluation_{timestamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = report["summary"]
    print(f"自动检查：{summary['passed']} 通过，{summary['failed']} 失败")
    print(f"人工审核：{summary['human_review_pending']} 个案例待检查")
    print(f"报告：{output}")


if __name__ == "__main__":
    main()
