"""离线记录模型回答的人工审核结论；不执行评估或调用模型。"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "evals" / "reports"
REVIEW_STATUSES = ("passed", "failed")
ALL_REVIEW_STATUSES = (*REVIEW_STATUSES, "pending")


def _reject_non_json_number(value: str) -> None:
    """拒绝 Python JSON 解码器默认接受的 NaN、Infinity。"""
    raise ValueError(f"输入报告含有非标准 JSON 数值：{value}")


def _load_report(report_path: Path) -> dict[str, Any]:
    report = json.loads(
        report_path.read_text(encoding="utf-8"),
        parse_constant=_reject_non_json_number,
    )
    if not isinstance(report, dict):
        raise ValueError("报告顶层必须是 JSON 对象")
    if not isinstance(report.get("cases"), list):
        raise ValueError("报告的 cases 必须是数组")
    if not isinstance(report.get("summary"), dict):
        raise ValueError("报告的 summary 必须是对象")
    return report


def _update_human_review(
    report: dict[str, Any], case_id: str, status: str, note: str
) -> dict[str, int]:
    """只更新目标案例人工审核和人工汇总，不修改原有自动检查指标。"""
    if status not in REVIEW_STATUSES:
        raise ValueError("审核状态只能是 passed 或 failed")
    if not isinstance(note, str) or not note.strip():
        raise ValueError("审核备注 --note 不能是空白字符串")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("案例 ID --case 不能为空")

    cases = report["cases"]
    if not all(isinstance(case, dict) for case in cases):
        raise ValueError("cases 中每个案例都必须是对象")
    matches = [case for case in cases if case.get("id") == case_id]
    if not matches:
        raise ValueError(f"未找到案例 ID：{case_id}")
    if len(matches) != 1:
        raise ValueError(f"案例 ID 重复（共 {len(matches)} 条）：{case_id}")

    review = matches[0].get("human_review")
    if not isinstance(review, dict):
        raise ValueError(f"案例 {case_id} 的 human_review 必须是对象")

    # 先验证所有审核状态，确保任何格式错误都不会产生输出文件。
    for case in cases:
        human_review = case.get("human_review")
        if not isinstance(human_review, dict):
            raise ValueError("每个案例都必须包含 human_review 对象")
        prior_status = human_review.get("status")
        if prior_status not in ALL_REVIEW_STATUSES:
            raise ValueError(f"案例 {case.get('id')} 的人工审核状态无效")

    review.update(
        {
            "status": status,
            "note": note.strip(),
            "reviewed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
    )
    counts = {
        "human_review_passed": sum(
            case["human_review"]["status"] == "passed" for case in cases
        ),
        "human_review_failed": sum(
            case["human_review"]["status"] == "failed" for case in cases
        ),
        "human_review_pending": sum(
            case["human_review"]["status"] == "pending" for case in cases
        ),
    }
    report["summary"].update(counts)
    return counts


def review_report(
    report_path: Path,
    case_id: str,
    status: str,
    note: str,
    *,
    output_dir: Path = DEFAULT_REPORT_DIR,
) -> tuple[Path, dict[str, int]]:
    """读取报告并写出新审核快照；输入报告和已有输出绝不覆盖。"""
    report = _load_report(Path(report_path))
    counts = _update_human_review(report, case_id, status, note)
    # 保存前严格序列化；失败时不会创建报告文件。
    payload = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = output_dir / f"review_{timestamp}_{uuid4().hex}.json"
    # x 为排他创建模式，哪怕碰巧撞名也不会覆写已有文件。
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    return output, counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="离线记录评估报告的人工审核结论")
    parser.add_argument("--report", type=Path, required=True, help="已有 JSON 报告路径")
    parser.add_argument("--case", required=True, help="需要人工审核的案例 ID")
    parser.add_argument("--status", choices=REVIEW_STATUSES, required=True)
    parser.add_argument("--note", required=True, help="非空白人工审核说明")
    args = parser.parse_args(argv)

    try:
        output, counts = review_report(args.report, args.case, args.status, args.note)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))

    print(f"新审核报告：{output}")
    print(
        "人工审核汇总："
        f"通过 {counts['human_review_passed']}，"
        f"失败 {counts['human_review_failed']}，"
        f"待审核 {counts['human_review_pending']}"
    )


if __name__ == "__main__":
    main()
