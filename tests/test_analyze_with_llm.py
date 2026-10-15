"""Offline checks for CSV -> Python statistics -> model explanation."""

import importlib.util
import json
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from openai import OpenAIError


@pytest.fixture
def workflow(monkeypatch):
    def forbid_network(*args, **kwargs):
        raise AssertionError("Tests must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    path = Path(__file__).resolve().parents[1] / "examples/analyze_with_llm.py"
    spec = importlib.util.spec_from_file_location("analyze_with_llm_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    answer = {
        "summary": "模拟解释",
        "explanation": "离线测试不评估模型质量",
        "missing_information": [],
    }
    ask = Mock(return_value={"answer": json.dumps(answer, ensure_ascii=False)})
    monkeypatch.setattr(module, "ask_llm", ask)
    return SimpleNamespace(module=module, ask=ask, answer=answer)


def sent_statistics(workflow):
    workflow.ask.assert_called_once()
    prompt = workflow.ask.call_args.args[0]
    # The example appends its statistics as the final JSON object in the prompt.
    return json.loads(prompt[prompt.index("{") :]), prompt


def printed_results(output):
    """Read the two displayed JSON objects, allowing human-readable headings."""
    decoder = json.JSONDecoder()
    results = []
    remaining = output
    while "{" in remaining:
        start = remaining.index("{")
        result, end = decoder.raw_decode(remaining[start:])
        results.append(result)
        remaining = remaining[start + end :]
    return results


def test_import_does_not_read_csv_or_call_model(workflow, monkeypatch):
    read = Mock(side_effect=AssertionError("Import must not read CSV"))
    ask = Mock(side_effect=AssertionError("Import must not call model"))
    monkeypatch.setattr("data_copilot.data_loader.load_csv", read)
    monkeypatch.setattr("data_copilot.llm_client.ask_llm", ask)
    spec = importlib.util.spec_from_file_location(
        "fresh_analysis_example", workflow.module.__file__
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    read.assert_not_called()
    ask.assert_not_called()


def test_real_sample_statistics_are_sent_from_any_working_directory(
    workflow, monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    workflow.module.main()
    statistics, prompt = sent_statistics(workflow)
    assert statistics == {
        "row_count": 40,
        "missing_values": {"sales": 2},
        "statistics": {
            "sales": {"count": 38, "mean": pytest.approx(694.6973684210526)}
        },
    }
    assert workflow.module.QUESTION in prompt
    assert "非缺失" in prompt and "补零" in prompt
    assert "null" in prompt
    assert "order_id" not in prompt
    assert "advertising_spend" not in prompt
    output = capsys.readouterr()
    assert output.err == ""
    assert printed_results(output.out) == [statistics, workflow.answer]
    json.dumps(statistics, allow_nan=False)


def test_only_necessary_statistics_are_sent_without_raw_records(
    workflow, monkeypatch, capsys
):
    df = pd.DataFrame(
        {"sales": [10.0, None, 30.0], "private": ["RAW_RECORD_CANARY"] * 3}
    )
    original = df.copy(deep=True)
    read = Mock(return_value=df)
    describe = Mock(wraps=workflow.module.describe_numeric)
    monkeypatch.setattr(workflow.module, "load_csv", read)
    monkeypatch.setattr(workflow.module, "describe_numeric", describe)
    workflow.module.main()
    statistics, prompt = sent_statistics(workflow)
    assert statistics == {
        "row_count": 3,
        "missing_values": {"sales": 1},
        "statistics": {"sales": {"count": 2, "mean": 20.0}},
    }
    assert "RAW_RECORD_CANARY" not in prompt
    assert "private" not in prompt
    describe.assert_called_once_with(df, "sales")
    read.assert_called_once()
    assert (
        read.call_args.args[0]
        == Path(workflow.module.__file__).resolve().parents[1]
        / "data/samples/sales.csv"
    )
    pd.testing.assert_frame_equal(df, original)
    assert printed_results(capsys.readouterr().out)[0] == statistics


def test_all_missing_sales_are_sent_as_null_not_zero(workflow, monkeypatch, capsys):
    df = pd.DataFrame({"sales": [float("nan"), float("nan")]})
    monkeypatch.setattr(workflow.module, "load_csv", Mock(return_value=df))
    workflow.module.main()
    statistics, _ = sent_statistics(workflow)
    assert statistics == {
        "row_count": 2,
        "missing_values": {"sales": 2},
        "statistics": {"sales": {"count": 0, "mean": None}},
    }
    assert printed_results(capsys.readouterr().out)[0] == statistics
    json.dumps(statistics, allow_nan=False)


@pytest.mark.parametrize(
    "stage",
    [
        "load_csv",
        "dataset_info",
        "describe_numeric",
        "ask_llm",
        "parse_analysis_answer",
    ],
)
def test_stage_failure_is_preserved_without_partial_report(
    workflow, monkeypatch, capsys, stage
):
    monkeypatch.setattr(
        workflow.module,
        "load_csv",
        Mock(return_value=pd.DataFrame({"sales": [10.0, 30.0]})),
    )
    parser = Mock(wraps=workflow.module.parse_analysis_answer)
    monkeypatch.setattr(workflow.module, "parse_analysis_answer", parser)
    original = (
        OpenAIError("offline API failure")
        if stage == "ask_llm"
        else RuntimeError(f"failure at {stage}")
    )
    monkeypatch.setattr(workflow.module, stage, Mock(side_effect=original))
    with pytest.raises(type(original)) as caught:
        workflow.module.main()
    assert caught.value is original
    output = capsys.readouterr()
    assert output.out == output.err == ""
    if stage in {"load_csv", "dataset_info", "describe_numeric"}:
        workflow.ask.assert_not_called()
    if stage != "parse_analysis_answer":
        parser.assert_not_called()


@pytest.mark.parametrize("values", [["not numeric"], [float("inf")]])
def test_invalid_numeric_data_never_reaches_model(
    workflow, monkeypatch, capsys, values
):
    monkeypatch.setattr(
        workflow.module, "load_csv", Mock(return_value=pd.DataFrame({"sales": values}))
    )
    with pytest.raises(ValueError):
        workflow.module.main()
    workflow.ask.assert_not_called()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("body", ["not JSON", '{"summary":"incomplete"}'])
def test_bad_model_answer_does_not_produce_a_report(
    workflow, monkeypatch, capsys, body
):
    monkeypatch.setattr(
        workflow.module,
        "load_csv",
        Mock(return_value=pd.DataFrame({"sales": [1.0, 2.0]})),
    )
    workflow.ask.return_value = {"answer": body}
    expected = json.JSONDecodeError if body == "not JSON" else ValueError
    with pytest.raises(expected):
        workflow.module.main()
    workflow.ask.assert_called_once()
    output = capsys.readouterr()
    assert output.out == output.err == ""
