"""Offline coverage of the JSON Schema experiment, without live requests."""

import importlib.util
import json
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai import OpenAIError


@pytest.fixture
def experiment(monkeypatch):
    """Load the example freshly and replace the locally imported SDK client."""

    def forbid_network(*args, **kwargs):
        raise AssertionError("Tests must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("DASHSCOPE_MODEL", "configured-model")
    dotenv = Mock()
    monkeypatch.setattr("dotenv.load_dotenv", dotenv)
    create = Mock(side_effect=AssertionError("No fake completion configured"))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    constructor = Mock(return_value=client)
    monkeypatch.setattr("openai.OpenAI", constructor)
    path = Path(__file__).resolve().parents[1] / "examples/structured_output.py"
    spec = importlib.util.spec_from_file_location("structured_output_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(
        module=module, constructor=constructor, create=create, dotenv=dotenv
    )


@pytest.fixture
def valid_answer():
    return {
        "summary": "信息不足",
        "explanation": "请提供数据",
        "missing_information": ["销售额数据"],
    }


def completion(content, *, finish_reason="stop", refusal=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, refusal=refusal),
            )
        ]
    )


def set_completion(experiment, value):
    experiment.create.side_effect = None
    experiment.create.return_value = value


def test_import_has_no_request_or_config_loading(experiment):
    experiment.constructor.assert_not_called()
    experiment.create.assert_not_called()
    experiment.dotenv.assert_not_called()


def test_schema_request_and_local_validation(
    experiment, valid_answer, monkeypatch, capsys
):
    body = json.dumps(valid_answer, ensure_ascii=False)
    set_completion(experiment, completion(body))
    parser = Mock(wraps=experiment.module.parse_analysis_answer)
    monkeypatch.setattr(experiment.module, "parse_analysis_answer", parser)
    result = experiment.module.ask_structured("请分析销售额")
    assert result == valid_answer
    assert json.loads(json.dumps(result, allow_nan=False)) == valid_answer
    parser.assert_called_once_with(body)
    experiment.constructor.assert_called_once_with(
        api_key="offline-key",
        base_url="https://example.invalid/v1",
        timeout=60.0,
    )
    experiment.create.assert_called_once()
    request = experiment.create.call_args.kwargs
    assert request["model"] == "configured-model"
    assert request["stream"] is False
    assert "stream_options" not in request
    assert request["messages"] == [
        {"role": "system", "content": experiment.module.SYSTEM_PROMPT},
        {"role": "user", "content": "请分析销售额"},
    ]
    response_format = request["response_format"]
    assert response_format["type"] == "json_schema"
    constraint = response_format["json_schema"]
    assert isinstance(constraint["name"], str) and constraint["name"].strip()
    assert constraint["strict"] is True
    schema = constraint["schema"]
    assert schema["type"] == "object"
    assert schema["properties"] == {
        "summary": {"type": "string"},
        "explanation": {"type": "string"},
        "missing_information": {"type": "array", "items": {"type": "string"}},
    }
    assert set(schema["required"]) == {"summary", "explanation", "missing_information"}
    assert schema["additionalProperties"] is False
    output = capsys.readouterr()
    assert output.out == output.err == ""


@pytest.mark.parametrize("question", ["", " \n\t"])
def test_blank_question_prevents_client_creation(experiment, question):
    with pytest.raises(ValueError, match="不能为空"):
        experiment.module.ask_structured(question)
    experiment.constructor.assert_not_called()
    experiment.create.assert_not_called()


@pytest.mark.parametrize(
    "name", ["DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "DASHSCOPE_MODEL"]
)
@pytest.mark.parametrize("value", [None, " "])
def test_missing_config(experiment, monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=name):
        experiment.module.ask_structured("question")
    experiment.constructor.assert_not_called()


def test_no_choices(experiment):
    set_completion(experiment, SimpleNamespace(choices=[]))
    with pytest.raises(RuntimeError, match="候选回答"):
        experiment.module.ask_structured("question")


@pytest.mark.parametrize(
    "reason,message",
    [
        ("length", "被截断"),
        (None, "未正常结束"),
        ("content_filter", "content_filter"),
        ("tool_calls", "tool_calls"),
    ],
)
def test_invalid_finish_prevents_parsing(experiment, monkeypatch, reason, message):
    set_completion(experiment, completion("not JSON", finish_reason=reason))
    parser = Mock()
    monkeypatch.setattr(experiment.module, "parse_analysis_answer", parser)
    with pytest.raises(RuntimeError, match=message):
        experiment.module.ask_structured("question")
    parser.assert_not_called()


def test_refusal_prevents_parsing(experiment, monkeypatch):
    set_completion(experiment, completion("not JSON", refusal="declined"))
    parser = Mock()
    monkeypatch.setattr(experiment.module, "parse_analysis_answer", parser)
    with pytest.raises(RuntimeError, match="拒绝回答"):
        experiment.module.ask_structured("question")
    parser.assert_not_called()


@pytest.mark.parametrize("content", [None, "", " \n\t"])
def test_empty_body(experiment, content):
    set_completion(experiment, completion(content))
    with pytest.raises(RuntimeError, match="有效正文"):
        experiment.module.ask_structured("question")


@pytest.mark.parametrize("body", ["not JSON", "```json\n{}\n```"])
def test_bad_json_is_not_repaired_or_retried(experiment, body):
    set_completion(experiment, completion(body))
    with pytest.raises(json.JSONDecodeError) as caught:
        experiment.module.ask_structured("question")
    assert caught.value.doc == body
    experiment.create.assert_called_once()


def test_blank_summary_rejected_despite_string_schema(experiment, valid_answer):
    valid_answer["summary"] = " \t"
    set_completion(experiment, completion(json.dumps(valid_answer)))
    with pytest.raises(ValueError, match="summary"):
        experiment.module.ask_structured("question")
    experiment.create.assert_called_once()


def test_api_failure_is_preserved_without_fallback(experiment):
    original = OpenAIError("schema request failed")
    experiment.create.side_effect = original
    with pytest.raises(OpenAIError) as caught:
        experiment.module.ask_structured("question")
    assert caught.value is original
    experiment.create.assert_called_once()
    assert (
        experiment.create.call_args.kwargs["response_format"]["type"] == "json_schema"
    )


def test_main_displays_each_question_result(
    experiment, valid_answer, monkeypatch, capsys
):
    questions = ("概念问题", "缺少数据的问题", "因果问题")
    monkeypatch.setattr(experiment.module, "QUESTIONS", questions)
    ask = Mock(return_value=valid_answer)
    monkeypatch.setattr(experiment.module, "ask_structured", ask)
    experiment.module.main()
    assert [call.args for call in ask.call_args_list] == [(q,) for q in questions]
    output = capsys.readouterr()
    assert output.err == ""
    # The CLI prints separate JSON documents, one per question.
    remaining = output.out.strip()
    decoder = json.JSONDecoder()
    records = []
    while remaining:
        record, end = decoder.raw_decode(remaining)
        records.append(record)
        remaining = remaining[end:].lstrip()
    assert records == [{"question": q, "result": valid_answer} for q in questions]
