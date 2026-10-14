"""Offline tests: fake credentials and responses; no API or real .env access."""

import importlib.util
import json
import runpy
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai import OpenAIError

from data_copilot import llm_client


@pytest.fixture(autouse=True)
def offline_client(monkeypatch):
    """Patch names where ask_llm looks them up, and forbid network connections."""

    def forbid_network(*args, **kwargs):
        raise AssertionError("Tests must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-test-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("DASHSCOPE_MODEL", "configured-model")
    dotenv = Mock()
    monkeypatch.setattr(llm_client, "load_dotenv", dotenv)
    create = Mock(side_effect=AssertionError("No fake response configured"))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    constructor = Mock(return_value=client)
    monkeypatch.setattr(llm_client, "OpenAI", constructor)
    return SimpleNamespace(constructor=constructor, create=create, dotenv=dotenv)


def chunk(
    content=None,
    finish_reason=None,
    *,
    usage=None,
    model="actual-model",
    reasoning=None,
    choices=True,
):
    """Only the response fields read by ask_llm are needed for a fake chunk."""
    delta = SimpleNamespace(content=content, reasoning_content=reasoning)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(
        model=model, usage=usage, choices=[choice] if choices else []
    )


def set_response(offline_client, chunks):
    offline_client.create.side_effect = None
    offline_client.create.return_value = iter(chunks)


def test_streaming_answer_and_usage(offline_client, capsys):
    usage = SimpleNamespace(prompt_tokens=7, completion_tokens=11, total_tokens=18)
    set_response(
        offline_client,
        [
            chunk(reasoning="private reasoning, do not return"),
            chunk("统计"),
            chunk("分析", reasoning="also ignored"),
            chunk(finish_reason="stop"),
            chunk(usage=usage, choices=False),
        ],
    )
    result = llm_client.ask_llm("请解释均值")
    assert result == {
        "answer": "统计分析",
        "model": "actual-model",
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18},
    }
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""
    offline_client.constructor.assert_called_once_with(
        api_key="offline-test-key",
        base_url="https://example.invalid/v1",
        timeout=60.0,
    )
    request = offline_client.create.call_args.kwargs
    assert request["model"] == "configured-model"
    assert request["messages"] == [{"role": "user", "content": "请解释均值"}]
    assert request["stream"] is True
    assert request["stream_options"] == {"include_usage": True}


@pytest.mark.parametrize("question", ["", " ", "\n\t", "　"])
def test_blank_question_does_not_create_client(offline_client, question):
    with pytest.raises(ValueError, match="不能为空"):
        llm_client.ask_llm(question)
    offline_client.constructor.assert_not_called()
    offline_client.create.assert_not_called()


@pytest.mark.parametrize(
    "name", ["DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "DASHSCOPE_MODEL"]
)
@pytest.mark.parametrize("value", [None, "", " \t"])
def test_missing_config(offline_client, monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=name):
        llm_client.ask_llm("question")
    offline_client.constructor.assert_not_called()


def test_all_missing_config_names_are_reported(offline_client, monkeypatch):
    names = ["DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "DASHSCOPE_MODEL"]
    for name in names:
        monkeypatch.delenv(name)
    with pytest.raises(RuntimeError) as caught:
        llm_client.ask_llm("question")
    assert all(name in str(caught.value) for name in names)
    offline_client.constructor.assert_not_called()


def test_missing_usage(offline_client):
    set_response(offline_client, [chunk("answer", "stop")])
    result = llm_client.ask_llm("another question")
    assert result["usage"] is None
    assert result["answer"] == "answer"
    assert json.loads(json.dumps(result, allow_nan=False))["usage"] is None
    assert offline_client.create.call_args.kwargs["messages"] == [
        {"role": "user", "content": "another question"},
    ]


def test_missing_model_is_not_replaced_with_configured_model(offline_client):
    set_response(offline_client, [chunk("answer", "stop", model=None)])
    assert llm_client.ask_llm("question")["model"] is None


@pytest.mark.parametrize(
    "finish_reason,message",
    [
        ("length", "被截断"),
        (None, "未收到生成结束标记"),
        ("content_filter", "content_filter"),
        ("tool_calls", "tool_calls"),
    ],
)
def test_abnormal_finish(offline_client, finish_reason, message):
    set_response(offline_client, [chunk("partial answer", finish_reason)])
    with pytest.raises(RuntimeError, match=message):
        llm_client.ask_llm("question")


@pytest.mark.parametrize("content", [None, "", " \n\t"])
def test_blank_body_even_when_reasoning_exists(offline_client, content):
    set_response(
        offline_client,
        [
            chunk(reasoning="reasoning is not an answer"),
            chunk(content, "stop"),
        ],
    )
    with pytest.raises(RuntimeError, match="有效正文"):
        llm_client.ask_llm("question")


def test_empty_stream(offline_client):
    set_response(offline_client, [])
    with pytest.raises(RuntimeError, match="未收到生成结束标记"):
        llm_client.ask_llm("question")


def test_request_exception_is_preserved(offline_client, capsys):
    original = OpenAIError("offline request failure")
    offline_client.create.side_effect = original
    with pytest.raises(OpenAIError) as caught:
        llm_client.ask_llm("question")
    assert caught.value is original
    output = capsys.readouterr()
    assert output.out == output.err == ""


def test_stream_exception_is_preserved(offline_client, capsys):
    original = OpenAIError("offline stream failure")

    def interrupted_stream():
        yield chunk("partial answer")
        raise original

    set_response(offline_client, interrupted_stream())
    with pytest.raises(OpenAIError) as caught:
        llm_client.ask_llm("question")
    assert caught.value is original
    output = capsys.readouterr()
    assert output.out == output.err == ""


def test_import_does_not_load_config_or_create_client(monkeypatch):
    constructor = Mock(side_effect=AssertionError("Import must not create a client"))
    dotenv = Mock(side_effect=AssertionError("Import must not read .env"))
    monkeypatch.setattr("openai.OpenAI", constructor)
    monkeypatch.setattr("dotenv.load_dotenv", dotenv)
    # Load a fresh module without reloading the shared module patched by fixtures.
    spec = importlib.util.spec_from_file_location(
        "isolated_llm_client", llm_client.__file__
    )
    assert spec is not None and spec.loader is not None
    isolated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(isolated)
    assert callable(isolated.ask_llm)
    constructor.assert_not_called()
    dotenv.assert_not_called()


def test_example_delegates_and_displays_only_result(monkeypatch, capsys):
    result = {
        "answer": "测试回复",
        "model": "actual-model",
        "finish_reason": "stop",
        "usage": None,
    }
    ask = Mock(return_value=result)
    monkeypatch.setattr(llm_client, "ask_llm", ask)
    path = Path(__file__).resolve().parents[1] / "examples/hello_llm.py"
    runpy.run_path(str(path), run_name="__main__")
    ask.assert_called_once()
    question = ask.call_args.args[0]
    assert isinstance(question, str) and question.strip()
    output = capsys.readouterr()
    assert json.loads(output.out) == result
    assert output.err == ""
