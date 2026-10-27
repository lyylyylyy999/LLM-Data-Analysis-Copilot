"""Offline manual-review snapshots: immutable input and independent metrics."""

import copy
import importlib.util
import json
import socket
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def reviewer(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("Review must not request a model"))
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr("os.getenv", forbidden)
    path = Path(__file__).resolve().parents[1] / "examples/review_evaluation.py"
    spec = importlib.util.spec_from_file_location("review_evaluation_offline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    forbidden.assert_not_called()
    return module


@pytest.fixture
def report(tmp_path):
    payload = {
        "generated_at": "2026-10-26T12:36:44Z",
        "summary": {"total": 3, "passed": 2, "failed": 1, "human_review_pending": 3},
        "cases": [
            {
                "id": identifier,
                "model": "actual-model",
                "answer": f"回答 {identifier}",
                "tool_results": [{"result": {"value": None}}],
                "automatic_check": {"passed": identifier != "c", "requirements": []},
                "error_type": None,
                "human_review": {
                    "status": "pending",
                    "criteria": [f"标准 {identifier}"],
                },
            }
            for identifier in ("a", "b", "c")
        ],
    }
    path = tmp_path / "source.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return SimpleNamespace(path=path, payload=payload, output=tmp_path / "reviews")


def review(reviewer, report, **kwargs):
    arguments = {
        "case_id": "a",
        "status": "passed",
        "note": "审核说明",
        "output_dir": report.output,
    }
    arguments.update(kwargs)
    return reviewer.review_report(report.path, **arguments)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_review_preserves_source_content_and_unrelated_fields(reviewer, report):
    source = report.path.read_bytes()
    output, counts = review(reviewer, report)
    assert output != report.path and output.parent == report.output
    assert report.path.read_bytes() == source
    actual = read(output)
    assert actual["generated_at"] == report.payload["generated_at"]
    assert actual["cases"][1:] == report.payload["cases"][1:]
    for field in ("answer", "model", "tool_results", "automatic_check", "error_type"):
        assert actual["cases"][0][field] == report.payload["cases"][0][field]
    human = actual["cases"][0]["human_review"]
    assert human["criteria"] == ["标准 a"]
    assert human["status"] == "passed" and human["note"] == "审核说明"
    assert datetime.fromisoformat(human["reviewed_at"]).utcoffset().total_seconds() == 0
    assert counts == {
        "human_review_passed": 1,
        "human_review_failed": 0,
        "human_review_pending": 2,
    }
    assert {key: actual["summary"][key] for key in ("total", "passed", "failed")} == {
        "total": 3,
        "passed": 2,
        "failed": 1,
    }
    json.dumps(actual, ensure_ascii=False, allow_nan=False)


def test_sequential_reviews_accumulate_without_changing_prior_snapshot(
    reviewer, report
):
    first, _ = review(reviewer, report)
    first_bytes = first.read_bytes()
    second, counts = reviewer.review_report(
        first, "b", "failed", "回答解释不准确", output_dir=report.output
    )
    assert first != second
    assert first.read_bytes() == first_bytes
    assert read(second)["cases"][0] == read(first)["cases"][0]
    assert read(second)["cases"][1]["human_review"]["status"] == "failed"
    assert counts == {
        "human_review_passed": 1,
        "human_review_failed": 1,
        "human_review_pending": 1,
    }
    assert read(second)["summary"]["passed"] == 2
    assert read(second)["summary"]["failed"] == 1


def test_revising_same_case_updates_counts_instead_of_double_counting(reviewer, report):
    first, _ = review(reviewer, report)
    output, counts = reviewer.review_report(
        first, "a", "failed", "复核后发现错误", output_dir=report.output
    )
    assert counts == {
        "human_review_passed": 0,
        "human_review_failed": 1,
        "human_review_pending": 2,
    }
    assert read(output)["cases"][0]["human_review"]["note"] == "复核后发现错误"
    assert read(first)["cases"][0]["human_review"]["status"] == "passed"


@pytest.mark.parametrize("case_id,status", [("a", "failed"), ("c", "passed")])
def test_human_decision_does_not_follow_automatic_check(
    reviewer, report, case_id, status
):
    output, _ = review(reviewer, report, case_id=case_id, status=status)
    actual = read(output)
    assert actual["summary"]["passed"] == 2 and actual["summary"]["failed"] == 1
    record = next(case for case in actual["cases"] if case["id"] == case_id)
    assert record["human_review"]["status"] == status
    assert record["automatic_check"]["passed"] is (case_id != "c")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"case_id": "absent"},
        {"case_id": ""},
        {"case_id": " \t"},
        {"status": "pending"},
        {"status": "PASS"},
        {"status": ""},
        {"note": ""},
        {"note": " \n\t"},
        {"note": None},
    ],
)
def test_invalid_review_does_not_create_output_or_change_input(
    reviewer, report, kwargs
):
    source = report.path.read_bytes()
    with pytest.raises(ValueError):
        review(reviewer, report, **kwargs)
    assert report.path.read_bytes() == source
    assert not report.output.exists()


def test_duplicate_target_id_is_rejected(reviewer, report):
    payload = copy.deepcopy(report.payload)
    payload["cases"].append(copy.deepcopy(payload["cases"][0]))
    report.path.write_text(json.dumps(payload), encoding="utf-8")
    before = report.path.read_bytes()
    with pytest.raises(ValueError, match="重复"):
        review(reviewer, report)
    assert report.path.read_bytes() == before
    assert not report.output.exists()


@pytest.mark.parametrize(
    "mutation", ["missing_review", "invalid_status", "nonobject_case"]
)
def test_malformed_other_case_prevents_writing(reviewer, report, mutation):
    payload = copy.deepcopy(report.payload)
    if mutation == "missing_review":
        del payload["cases"][1]["human_review"]
    elif mutation == "invalid_status":
        payload["cases"][1]["human_review"]["status"] = "unknown"
    else:
        payload["cases"][1] = None
    report.path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        review(reviewer, report)
    assert not report.output.exists()


@pytest.mark.parametrize(
    "text",
    [
        "{broken",
        "[]",
        '{"cases":[],"summary":[]}',
        '{"cases":{},"summary":{}}',
        '{"cases":[],"summary":{},"invalid":NaN}',
        '{"cases":[],"summary":{},"invalid":Infinity}',
    ],
)
def test_invalid_report_is_rejected_before_output(reviewer, report, text):
    report.path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        review(reviewer, report)
    assert not report.output.exists()


def test_numeric_overflow_is_rejected_before_output(reviewer, report):
    text = json.dumps(report.payload).replace('"value": null', '"value": 1e999')
    report.path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        review(reviewer, report)
    assert not report.output.exists()


def test_filename_collision_does_not_overwrite_existing_report(
    reviewer, report, monkeypatch
):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 27, 1, 2, 3, tzinfo=UTC)

    monkeypatch.setattr(reviewer, "datetime", FixedDateTime)
    monkeypatch.setattr(reviewer, "uuid4", lambda: SimpleNamespace(hex="fixed"))
    report.output.mkdir()
    existing = report.output / "review_20261027T010203000000Z_fixed.json"
    existing.write_text("existing report", encoding="utf-8")
    with pytest.raises(FileExistsError):
        review(reviewer, report)
    assert existing.read_text(encoding="utf-8") == "existing report"
    assert list(report.output.iterdir()) == [existing]
    assert read(report.path) == report.payload


def test_cli_delegates_and_prints_manual_counts(reviewer, report, monkeypatch, capsys):
    counts = {
        "human_review_passed": 1,
        "human_review_failed": 0,
        "human_review_pending": 2,
    }
    write = Mock(return_value=(report.output / "review.json", counts))
    monkeypatch.setattr(reviewer, "review_report", write)
    reviewer.main(
        [
            "--report",
            str(report.path),
            "--case",
            "a",
            "--status",
            "passed",
            "--note",
            "审核通过",
        ]
    )
    write.assert_called_once_with(report.path, "a", "passed", "审核通过")
    output = capsys.readouterr().out
    assert "通过 1，失败 0，待审核 2" in output
    assert "review.json" in output


def test_cli_failure_exits_without_creating_output(reviewer, report, capsys):
    with pytest.raises(SystemExit) as caught:
        reviewer.main(
            [
                "--report",
                str(report.path),
                "--case",
                "absent",
                "--status",
                "passed",
                "--note",
                "说明",
            ]
        )
    assert caught.value.code == 2
    assert "未找到案例" in capsys.readouterr().err


def test_missing_source_never_creates_output(reviewer, report):
    with pytest.raises(FileNotFoundError):
        reviewer.review_report(
            report.path.parent / "absent.json",
            "a",
            "passed",
            "说明",
            output_dir=report.output,
        )
    assert not report.output.exists()
