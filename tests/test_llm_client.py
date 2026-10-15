"""Offline tests: fake credentials and responses; no API or real .env access."""

import importlib.util
import json
import runpy
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from openai import OpenAI, OpenAIError
from pydantic import ValidationError

from data_copilot import llm_client
from data_copilot.schemas import AnalysisAnswer

ANSWER = json.dumps(
    {"summary": "统计分析", "explanation": "", "missing_information": []},
    ensure_ascii=False,
)


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
            chunk(ANSWER[:15]),
            chunk(ANSWER[15:], reasoning="also ignored"),
            chunk(finish_reason="stop"),
            chunk(usage=usage, choices=False),
        ],
    )
    result = llm_client.ask_llm("请解释均值")
    assert result == {
        "answer": ANSWER,
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
    assert llm_client.SYSTEM_PROMPT.strip()
    assert request["messages"] == [
        {"role": "system", "content": llm_client.SYSTEM_PROMPT},
        {"role": "user", "content": "请解释均值"},
    ]
    assert request["stream"] is True
    assert request["stream_options"] == {"include_usage": True}
    assert request["extra_body"] == {"enable_thinking": False}
    assert request["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "AnalysisAnswer",
            "strict": True,
            "schema": AnalysisAnswer.model_json_schema(),
        },
    }


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
    set_response(offline_client, [chunk(ANSWER, "stop")])
    result = llm_client.ask_llm("another question")
    assert result["usage"] is None
    assert result["answer"] == ANSWER
    assert json.loads(json.dumps(result, allow_nan=False))["usage"] is None
    assert offline_client.create.call_args.kwargs["messages"] == [
        {"role": "system", "content": llm_client.SYSTEM_PROMPT},
        {"role": "user", "content": "another question"},
    ]


def test_missing_model_is_not_replaced_with_configured_model(offline_client):
    set_response(offline_client, [chunk(ANSWER, "stop", model=None)])
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


@pytest.mark.parametrize(
    "answer",
    [
        "not JSON",
        "{}",
        "[]",
        '{"summary":" ","explanation":"","missing_information":[]}',
        '{"summary":1,"explanation":"","missing_information":[]}',
        '{"summary":"ok","explanation":"","missing_information":[1]}',
        '{"summary":"ok","explanation":"","missing_information":[],"extra":1}',
    ],
)
def test_invalid_structured_answer_is_not_returned(offline_client, answer):
    set_response(offline_client, [chunk(answer, "stop")])
    with pytest.raises(ValidationError):
        llm_client.ask_llm("question")
    offline_client.create.assert_called_once()


def test_refusal_is_not_returned_as_an_answer(offline_client):
    refusal = chunk()
    refusal.choices[0].delta.refusal = "declined"
    set_response(offline_client, [refusal, chunk(ANSWER, "stop")])
    with pytest.raises(RuntimeError, match="拒绝回答"):
        llm_client.ask_llm("question")


def test_real_sdk_sends_schema_and_reads_stream(monkeypatch):
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        events = [
            {
                "id": "offline-stream",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "actual-model",
                "choices": [
                    {"index": 0, "delta": {"content": ANSWER}, "finish_reason": None}
                ],
            },
            {
                "id": "offline-stream",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "actual-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 11,
                    "total_tokens": 18,
                },
            },
        ]
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=body + "data: [DONE]\n\n",
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
        with OpenAI(
            api_key="offline-key",
            base_url="https://example.invalid/v1",
            max_retries=0,
            http_client=http_client,
        ) as client:
            monkeypatch.setattr(llm_client, "OpenAI", Mock(return_value=client))
            result = llm_client.ask_llm("question")

    assert result == {
        "answer": ANSWER,
        "model": "actual-model",
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18},
    }
    assert len(requests) == 1
    request = requests[0]
    assert request["stream"] is True
    assert request["enable_thinking"] is False
    assert request["response_format"]["type"] == "json_schema"
    schema = request["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"] == AnalysisAnswer.model_json_schema()


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


def test_example_delegates_and_displays_parsed_answer(monkeypatch, capsys):
    parsed = {
        "summary": "测试回复",
        "explanation": "测试解释",
        "missing_information": [],
    }
    result = {
        "answer": json.dumps(parsed, ensure_ascii=False),
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
    assert json.loads(output.out) == parsed
    assert output.err == ""


def test_example_does_not_hide_json_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        llm_client, "ask_llm", Mock(return_value={"answer": "not JSON"})
    )
    path = Path(__file__).resolve().parents[1] / "examples/hello_llm.py"
    with pytest.raises(json.JSONDecodeError) as caught:
        runpy.run_path(str(path), run_name="__main__")
    assert caught.value.doc == "not JSON"
    output = capsys.readouterr()
    assert output.out == output.err == ""


@pytest.mark.parametrize(
    "question",
    [
        "什么是过拟合？请用一个简单例子解释。",
        "销售额和广告投入正相关，能证明广告提高了销售额吗？",
        "请告诉我这份数据的平均销售额。",
    ],
)
def test_system_rules_and_question_are_separate(offline_client, monkeypatch, question):
    """Verify message construction only; fake answers cannot evaluate the prompt."""
    rules = "测试系统规则：使用中文回答，不编造统计结果。"
    monkeypatch.setattr(llm_client, "SYSTEM_PROMPT", rules)
    set_response(offline_client, [chunk(ANSWER, "stop")])
    result = llm_client.ask_llm(question)
    offline_client.create.assert_called_once()
    assert offline_client.create.call_args.kwargs["messages"] == [
        {"role": "system", "content": rules},
        {"role": "user", "content": question},
    ]
    assert result == {
        "answer": ANSWER,
        "model": "actual-model",
        "finish_reason": "stop",
        "usage": None,
    }
