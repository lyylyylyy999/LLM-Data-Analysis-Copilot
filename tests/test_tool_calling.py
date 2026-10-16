"""Observe tool-call requests offline; never execute tools or contact an API."""

import importlib.util
import json
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai import OpenAIError


@pytest.fixture
def example(monkeypatch):
    def forbid_network(*args, **kwargs):
        raise AssertionError("Tests must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("DASHSCOPE_MODEL", "configured-model")
    create = Mock(side_effect=AssertionError("No fake response configured"))
    constructor = Mock(
        return_value=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )
    )
    dotenv = Mock()
    monkeypatch.setattr("openai.OpenAI", constructor)
    monkeypatch.setattr("dotenv.load_dotenv", dotenv)
    execute = Mock(side_effect=AssertionError("Task 08 must not execute dataset_info"))
    monkeypatch.setattr("data_copilot.analysis_tools.dataset_info", execute)
    monkeypatch.setattr("data_copilot.data_loader.dataset_info", execute)
    read = Mock(side_effect=AssertionError("Task 08 must not read a dataset"))
    monkeypatch.setattr("data_copilot.data_loader.load_csv", read)
    path = Path(__file__).resolve().parents[1] / "examples/tool_calling.py"
    spec = importlib.util.spec_from_file_location("tool_calling_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(
        module=module,
        constructor=constructor,
        create=create,
        dotenv=dotenv,
        execute=execute,
        read=read,
    )


def tool_call(identifier="call_1", arguments="{}"):
    return SimpleNamespace(
        id=identifier,
        type="function",
        function=SimpleNamespace(name="dataset_info", arguments=arguments),
    )


def set_response(example, calls=None, *, content=None, finish_reason="tool_calls"):
    example.create.side_effect = None
    example.create.return_value = SimpleNamespace(
        model="actual-model",
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(tool_calls=calls, content=content),
            )
        ],
    )


def test_import_has_no_request_execution_or_output(example, capsys):
    example.constructor.assert_not_called()
    example.create.assert_not_called()
    example.dotenv.assert_not_called()
    example.execute.assert_not_called()
    example.read.assert_not_called()
    output = capsys.readouterr()
    assert output.out == output.err == ""


def test_request_describes_empty_argument_tool_and_auto_choice(example, capsys):
    set_response(example, [tool_call()])
    example.module.main()
    example.constructor.assert_called_once_with(
        api_key="offline-key",
        base_url="https://example.invalid/v1",
        timeout=60.0,
    )
    example.create.assert_called_once()
    request = example.create.call_args.kwargs
    assert request["model"] == "configured-model"
    assert request["stream"] is False
    assert request["tool_choice"] == "auto"
    assert "response_format" not in request
    assert "stream_options" not in request
    assert request["messages"][1] == {
        "role": "user",
        "content": example.module.QUESTION,
    }
    assert request["messages"][0]["role"] == "system"
    assert request["messages"][0]["content"].strip()
    assert len(request["tools"]) == 1
    definition = request["tools"][0]
    assert definition["type"] == "function"
    function = definition["function"]
    assert function["name"] == "dataset_info"
    assert isinstance(function["description"], str) and function["description"].strip()
    assert function["parameters"] == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    json.dumps(request, allow_nan=False)
    output = capsys.readouterr()
    assert output.err == ""
    for text in [
        "actual-model",
        "tool_calls",
        "call_1",
        "dataset_info",
        "原始 arguments：{}",
        "解析后参数（dict）：{}",
        "尚未执行工具",
    ]:
        assert text in output.out
    example.execute.assert_not_called()
    example.read.assert_not_called()


def test_multiple_calls_are_displayed_without_executing_or_requesting_again(
    example, capsys
):
    set_response(example, [tool_call("first"), tool_call("second")])
    example.module.main()
    output = capsys.readouterr().out
    assert "工具调用 1" in output and "工具调用 2" in output
    assert "first" in output and "second" in output
    assert output.count("解析后参数（dict）：{}") == 2
    example.create.assert_called_once()
    example.execute.assert_not_called()
    example.read.assert_not_called()


@pytest.mark.parametrize("calls", [None, []])
@pytest.mark.parametrize("content", [None, "需要更多信息"])
def test_no_tool_request_is_reported_honestly(example, capsys, calls, content):
    set_response(example, calls, content=content, finish_reason="stop")
    example.module.main()
    output = capsys.readouterr()
    assert output.err == ""
    assert "本次没有请求工具" in output.out
    assert "工具调用 1" not in output.out
    if content:
        assert content in output.out
    example.create.assert_called_once()
    example.execute.assert_not_called()


def test_no_choices(example):
    example.create.side_effect = None
    example.create.return_value = SimpleNamespace(choices=[])
    with pytest.raises(RuntimeError, match="choice"):
        example.module.main()


def test_truncation_is_not_treated_as_complete_tool_request(example, capsys):
    set_response(example, [tool_call()], finish_reason="length")
    with pytest.raises(RuntimeError, match="被截断"):
        example.module.main()
    assert capsys.readouterr().out == ""
    example.execute.assert_not_called()


@pytest.mark.parametrize("arguments", ["", "not JSON", '{"incomplete":'])
def test_invalid_json_arguments_preserve_decode_error(example, arguments):
    set_response(example, [tool_call(arguments=arguments)])
    with pytest.raises(json.JSONDecodeError) as caught:
        example.module.main()
    assert caught.value.doc == arguments
    example.execute.assert_not_called()
    example.create.assert_called_once()


@pytest.mark.parametrize("value", [[], None, 1, "text", True])
def test_arguments_must_be_json_object(example, value):
    set_response(example, [tool_call(arguments=json.dumps(value))])
    with pytest.raises(ValueError, match="JSON 对象"):
        example.module.main()
    example.execute.assert_not_called()


@pytest.mark.parametrize(
    "name", ["DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "DASHSCOPE_MODEL"]
)
@pytest.mark.parametrize("value", [None, " \t"])
def test_missing_config_prevents_request(example, monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=name):
        example.module.main()
    example.constructor.assert_not_called()
    example.create.assert_not_called()


def test_api_error_propagates_unchanged_without_second_request(example):
    original = OpenAIError("offline tool request failure")
    example.create.side_effect = original
    with pytest.raises(OpenAIError) as caught:
        example.module.main()
    assert caught.value is original
    example.create.assert_called_once()
    example.execute.assert_not_called()
