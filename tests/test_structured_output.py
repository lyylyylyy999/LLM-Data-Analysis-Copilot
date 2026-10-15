"""Offline tests of SDK-generated Schema and Pydantic response parsing."""

import importlib.util
import json
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from openai import LengthFinishReasonError, OpenAI, OpenAIError
from pydantic import ValidationError

from data_copilot.schemas import AnalysisAnswer


@pytest.fixture
def experiment(monkeypatch):
    """Load the example freshly and replace its locally imported SDK client."""

    def forbid_network(*args, **kwargs):
        raise AssertionError("Tests must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("DASHSCOPE_MODEL", "configured-model")
    dotenv = Mock()
    monkeypatch.setattr("dotenv.load_dotenv", dotenv)
    parse = Mock(side_effect=AssertionError("No fake completion configured"))
    create = Mock(side_effect=AssertionError("Must not fall back to create()"))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(parse=parse, create=create))
    )
    constructor = Mock(return_value=client)
    monkeypatch.setattr("openai.OpenAI", constructor)
    path = Path(__file__).resolve().parents[1] / "examples/structured_output.py"
    spec = importlib.util.spec_from_file_location("structured_output_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(
        module=module,
        constructor=constructor,
        parse=parse,
        create=create,
        dotenv=dotenv,
    )


@pytest.fixture
def valid_answer():
    return {
        "summary": "信息不足",
        "explanation": "请提供数据",
        "missing_information": ["销售额数据"],
    }


def completion(parsed, *, finish_reason="stop", refusal=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(parsed=parsed, refusal=refusal),
            )
        ]
    )


def set_completion(experiment, value):
    experiment.parse.side_effect = None
    experiment.parse.return_value = value


def test_import_has_no_request_or_config_loading(experiment):
    experiment.constructor.assert_not_called()
    experiment.parse.assert_not_called()
    experiment.create.assert_not_called()
    experiment.dotenv.assert_not_called()


def test_model_class_is_sent_and_parsed_model_is_dumped(
    experiment, valid_answer, capsys
):
    model = AnalysisAnswer.model_validate(valid_answer)
    set_completion(experiment, completion(model))
    result = experiment.module.ask_structured("请分析销售额")
    assert isinstance(result, dict)
    assert result == valid_answer
    assert json.loads(json.dumps(result, allow_nan=False)) == valid_answer
    experiment.constructor.assert_called_once_with(
        api_key="offline-key",
        base_url="https://example.invalid/v1",
        timeout=60.0,
    )
    experiment.parse.assert_called_once()
    request = experiment.parse.call_args.kwargs
    assert request["model"] == "configured-model"
    assert request["response_format"] is AnalysisAnswer
    assert "stream_options" not in request
    assert request.get("stream", False) is False
    assert request["messages"] == [
        {"role": "system", "content": experiment.module.SYSTEM_PROMPT},
        {"role": "user", "content": "请分析销售额"},
    ]
    experiment.create.assert_not_called()
    output = capsys.readouterr()
    assert output.out == output.err == ""


@pytest.mark.parametrize("question", ["", " \n\t"])
def test_blank_question_prevents_client_creation(experiment, question):
    with pytest.raises(ValueError, match="不能为空"):
        experiment.module.ask_structured(question)
    experiment.constructor.assert_not_called()
    experiment.parse.assert_not_called()


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
def test_invalid_finish_is_rejected(experiment, reason, message):
    set_completion(experiment, completion(None, finish_reason=reason))
    with pytest.raises(RuntimeError, match=message):
        experiment.module.ask_structured("question")


def test_refusal_is_rejected(experiment):
    set_completion(experiment, completion(None, refusal="declined"))
    with pytest.raises(RuntimeError, match="拒绝回答"):
        experiment.module.ask_structured("question")


@pytest.mark.parametrize("parsed", [None, {}, "not a validated model"])
def test_missing_or_wrong_parsed_result_is_rejected(experiment, parsed):
    set_completion(experiment, completion(parsed))
    with pytest.raises(RuntimeError, match="有效解析结果"):
        experiment.module.ask_structured("question")


def test_sdk_error_is_preserved_without_fallback(experiment):
    original = OpenAIError("schema request failed")
    experiment.parse.side_effect = original
    with pytest.raises(OpenAIError) as caught:
        experiment.module.ask_structured("question")
    assert caught.value is original
    experiment.parse.assert_called_once()
    experiment.create.assert_not_called()


@pytest.mark.parametrize(
    "raw", ["not JSON", '{"summary":" ","explanation":"","missing_information":[]}']
)
def test_pydantic_error_is_preserved_without_fallback(experiment, raw):
    with pytest.raises(ValidationError) as initial:
        AnalysisAnswer.model_validate_json(raw)
    original = initial.value
    experiment.parse.side_effect = original
    with pytest.raises(ValidationError) as caught:
        experiment.module.ask_structured("question")
    assert caught.value is original
    experiment.parse.assert_called_once()
    experiment.create.assert_not_called()


def test_main_displays_single_question_result(
    experiment, valid_answer, monkeypatch, capsys
):
    # The example's main now demonstrates one request rather than a QUESTIONS loop.
    ask = Mock(return_value=valid_answer)
    monkeypatch.setattr(experiment.module, "ask_structured", ask)
    experiment.module.main()
    ask.assert_called_once_with("什么是过拟合？")
    output = capsys.readouterr()
    assert output.err == ""
    assert json.loads(output.out) == {
        "question": "什么是过拟合？",
        "result": valid_answer,
    }


@pytest.mark.parametrize(
    "scenario", ["valid", "blank_summary", "invalid_json", "length", "refusal"]
)
def test_real_sdk_with_mock_http_transport(
    experiment, valid_answer, monkeypatch, scenario
):
    """Use the actual SDK parser; MockTransport handles HTTP entirely in memory."""
    requests = []
    data = dict(valid_answer)
    if scenario == "blank_summary":
        data["summary"] = " \t"
    content = "not JSON" if scenario == "invalid_json" else json.dumps(data)

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "offline-completion",
                "object": "chat.completion",
                "created": 0,
                "model": "configured-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length" if scenario == "length" else "stop",
                        "message": {
                            "role": "assistant",
                            "content": content,
                            "refusal": "declined" if scenario == "refusal" else None,
                        },
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
        with OpenAI(
            api_key="offline-key",
            base_url="https://example.invalid/v1",
            max_retries=0,
            http_client=http_client,
        ) as client:
            monkeypatch.setattr("openai.OpenAI", Mock(return_value=client))
            if scenario == "valid":
                assert experiment.module.ask_structured("question") == valid_answer
            elif scenario in {"blank_summary", "invalid_json"}:
                with pytest.raises(ValidationError):
                    experiment.module.ask_structured("question")
            elif scenario == "length":
                with pytest.raises(LengthFinishReasonError):
                    experiment.module.ask_structured("question")
            else:
                with pytest.raises(RuntimeError, match="拒绝回答"):
                    experiment.module.ask_structured("question")

    assert len(requests) == 1
    response_format = requests[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert set(schema["required"]) == {"summary", "explanation", "missing_information"}
    assert schema["additionalProperties"] is False
    assert schema["properties"]["summary"]["type"] == "string"
    assert schema["properties"]["missing_information"]["items"]["type"] == "string"
