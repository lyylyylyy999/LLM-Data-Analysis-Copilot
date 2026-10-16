"""Offline validation of one tool execution and its two-request message chain."""

import copy
import importlib.util
import json
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from openai import OpenAIError
from openai.types.chat import ChatCompletionMessageFunctionToolCall

from data_copilot.analysis_tools import dataset_info as real_dataset_info
from data_copilot.data_loader import load_csv as real_load_csv


def tool_call(*, name="dataset_info", identifier="call_1", arguments="{}"):
    return ChatCompletionMessageFunctionToolCall(
        id=identifier,
        type="function",
        function={"name": name, "arguments": arguments},
    )


def response(*, calls=None, content=None, finish_reason="stop", refusal=None):
    return SimpleNamespace(
        model="actual-model",
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    tool_calls=calls, content=content, refusal=refusal
                ),
            )
        ],
    )


@pytest.fixture
def workflow(monkeypatch, tmp_path):
    def forbid_network(*args, **kwargs):
        raise AssertionError("Tests must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("DASHSCOPE_MODEL", "configured-model")
    dotenv = Mock()
    monkeypatch.setattr("dotenv.load_dotenv", dotenv)
    pending = [
        response(calls=[tool_call()], finish_reason="tool_calls"),
        response(content="数据有3行、2列。"),
    ]
    requests = []
    events = []

    def execute(dataframe):
        events.append("execute")
        return real_dataset_info(dataframe)

    execute_mock = Mock(side_effect=execute)

    def submit(**kwargs):
        # Snapshot at call time: the example extends the same messages list later.
        requests.append(copy.deepcopy(kwargs))
        events.append("request")
        if not pending:
            raise AssertionError("Unexpected extra model request")
        result = pending.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    create = Mock(side_effect=submit)
    constructor = Mock(
        return_value=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
    )
    monkeypatch.setattr("openai.OpenAI", constructor)
    path = Path(__file__).resolve().parents[1] / "examples/tool_execution.py"
    spec = importlib.util.spec_from_file_location("tool_execution_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    read = Mock(wraps=real_load_csv)
    monkeypatch.setattr(module, "load_csv", read)
    monkeypatch.setattr(module, "dataset_info", execute_mock)
    csv_path = tmp_path / "tiny.csv"
    csv_path.write_text("region,sales\nEast,10\nWest,\nNorth,30\n", encoding="utf-8")
    return SimpleNamespace(
        module=module,
        create=create,
        constructor=constructor,
        execute=execute_mock,
        read=read,
        pending=pending,
        requests=requests,
        events=events,
        dotenv=dotenv,
        csv=csv_path,
    )


def assert_no_execution_or_second_request(workflow):
    workflow.execute.assert_not_called()
    assert workflow.create.call_count == 1


def test_import_has_no_io(workflow):
    workflow.constructor.assert_not_called()
    workflow.create.assert_not_called()
    workflow.execute.assert_not_called()
    workflow.read.assert_not_called()
    workflow.dotenv.assert_not_called()


def test_complete_message_chain_uses_real_dataframe(workflow, capsys):
    # Preserve whitespace in the original arguments, not merely equivalent JSON.
    call = tool_call(identifier="original_call_id", arguments="{ }")
    workflow.pending[0] = response(
        calls=[call], content="正在获取数据集信息", finish_reason="tool_calls"
    )
    workflow.module.main(workflow.csv)
    workflow.read.assert_called_once_with(workflow.csv)
    workflow.constructor.assert_called_once_with(
        api_key="offline-key",
        base_url="https://example.invalid/v1",
        timeout=60.0,
    )
    workflow.execute.assert_called_once()
    dataframe = workflow.execute.call_args.args[0]
    pd.testing.assert_frame_equal(dataframe, real_load_csv(workflow.csv))
    expected = real_dataset_info(dataframe)
    assert expected["row_count"] == 3 and expected["column_count"] == 2
    assert expected["missing_values"]["sales"] == 1
    assert workflow.events == ["request", "execute", "request"]
    first, second = workflow.requests
    assert first["model"] == second["model"] == "configured-model"
    assert first["stream"] is second["stream"] is False
    assert first["tool_choice"] == "auto"
    assert first["tools"] == workflow.module.TOOLS
    assert len(first["messages"]) == 2
    assert first["messages"] == [
        {"role": "system", "content": workflow.module.SYSTEM_PROMPT},
        {"role": "user", "content": workflow.module.QUESTION},
    ]
    assert "tools" not in second and "tool_choice" not in second
    assert "response_format" not in first and "response_format" not in second
    assert second["messages"][:2] == first["messages"]
    assert len(second["messages"]) == 4
    assistant, tool = second["messages"][2:]
    assert assistant == {
        "role": "assistant",
        "content": "正在获取数据集信息",
        "tool_calls": [
            {
                "id": "original_call_id",
                "type": "function",
                "function": {"name": "dataset_info", "arguments": "{ }"},
            }
        ],
    }
    assert tool["role"] == "tool"
    assert tool["tool_call_id"] == "original_call_id"
    assert json.loads(tool["content"]) == expected
    json.dumps(second, allow_nan=False)
    output = capsys.readouterr()
    assert output.err == ""
    assert "original_call_id" in output.out
    assert "Python 实际执行的工具结果" in output.out
    assert "模型最终回答：数据有3行、2列。" in output.out


def test_repository_sample_uses_40_rows_and_6_columns(workflow, capsys):
    path = Path(__file__).resolve().parents[1] / "data/samples/sales.csv"
    workflow.pending[1] = response(content="数据有40行、6列。")
    workflow.module.main(path)
    result = json.loads(workflow.requests[1]["messages"][-1]["content"])
    assert result["row_count"] == 40
    assert result["column_count"] == 6
    assert result["missing_values"]["sales"] == 2
    assert "模型最终回答：数据有40行、6列。" in capsys.readouterr().out


@pytest.mark.parametrize("calls", [None, []])
def test_no_tool_request_stops_after_first_round(workflow, calls, capsys):
    workflow.pending[0] = response(calls=calls, content="需要先获取数据")
    workflow.module.main(workflow.csv)
    assert_no_execution_or_second_request(workflow)
    assert "本次没有请求工具" in capsys.readouterr().out


def test_no_first_choices(workflow):
    workflow.pending[0] = SimpleNamespace(choices=[])
    with pytest.raises(RuntimeError, match="第一轮.*choice"):
        workflow.module.main(workflow.csv)
    assert_no_execution_or_second_request(workflow)


@pytest.mark.parametrize("reason", ["length", None, "content_filter"])
def test_abnormal_first_finish_prevents_execution(workflow, reason):
    workflow.pending[0] = response(calls=[tool_call()], finish_reason=reason)
    with pytest.raises(RuntimeError):
        workflow.module.main(workflow.csv)
    assert_no_execution_or_second_request(workflow)


def test_first_refusal_prevents_execution(workflow):
    workflow.pending[0] = response(
        calls=[tool_call()], finish_reason="tool_calls", refusal="declined"
    )
    with pytest.raises(RuntimeError, match="第一轮.*拒绝"):
        workflow.module.main(workflow.csv)
    assert_no_execution_or_second_request(workflow)


@pytest.mark.parametrize(
    "calls,reason", [(None, "tool_calls"), ([tool_call()], "stop")]
)
def test_inconsistent_first_response_prevents_execution(workflow, calls, reason):
    workflow.pending[0] = response(calls=calls, finish_reason=reason)
    with pytest.raises(RuntimeError, match="tool_calls"):
        workflow.module.main(workflow.csv)
    assert_no_execution_or_second_request(workflow)


def test_multiple_calls_are_not_partially_executed(workflow):
    workflow.pending[0] = response(
        calls=[tool_call(), tool_call(identifier="call_2")], finish_reason="tool_calls"
    )
    with pytest.raises(ValueError, match="恰好调用一次"):
        workflow.module.main(workflow.csv)
    assert_no_execution_or_second_request(workflow)


def test_unknown_type_is_rejected_before_reading_function(workflow):
    workflow.pending[0] = response(
        calls=[SimpleNamespace(type="custom")], finish_reason="tool_calls"
    )
    with pytest.raises(ValueError, match="调用类型"):
        workflow.module.main(workflow.csv)
    assert_no_execution_or_second_request(workflow)


def test_unregistered_tool_is_rejected(workflow):
    workflow.pending[0] = response(
        calls=[tool_call(name="load_csv")], finish_reason="tool_calls"
    )
    with pytest.raises(ValueError, match="工具名称"):
        workflow.module.main(workflow.csv)
    assert_no_execution_or_second_request(workflow)


@pytest.mark.parametrize("identifier", [None, "", " \t"])
def test_empty_call_id_is_rejected(workflow, identifier):
    call = SimpleNamespace(
        type="function",
        id=identifier,
        function=SimpleNamespace(name="dataset_info", arguments="{}"),
    )
    workflow.pending[0] = response(calls=[call], finish_reason="tool_calls")
    with pytest.raises(ValueError, match="ID"):
        workflow.module.main(workflow.csv)
    assert_no_execution_or_second_request(workflow)


@pytest.mark.parametrize("arguments", ["", "not JSON", '{"incomplete":'])
def test_invalid_json_arguments_are_preserved(workflow, arguments):
    workflow.pending[0] = response(
        calls=[tool_call(arguments=arguments)], finish_reason="tool_calls"
    )
    with pytest.raises(json.JSONDecodeError) as caught:
        workflow.module.main(workflow.csv)
    assert caught.value.doc == arguments
    assert_no_execution_or_second_request(workflow)


@pytest.mark.parametrize(
    "value", [[], None, 1, True, "text", {"file_path": "other.csv"}]
)
def test_nonempty_or_nonobject_arguments_are_rejected(workflow, value):
    workflow.pending[0] = response(
        calls=[tool_call(arguments=json.dumps(value))], finish_reason="tool_calls"
    )
    with pytest.raises(ValueError):
        workflow.module.main(workflow.csv)
    assert_no_execution_or_second_request(workflow)


def test_missing_csv_prevents_any_request(workflow):
    with pytest.raises(FileNotFoundError):
        workflow.module.main(workflow.csv.parent / "absent.csv")
    workflow.create.assert_not_called()
    workflow.execute.assert_not_called()


@pytest.mark.parametrize(
    "name", ["DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "DASHSCOPE_MODEL"]
)
def test_missing_config_prevents_request(workflow, monkeypatch, name):
    monkeypatch.delenv(name)
    with pytest.raises(RuntimeError, match=name):
        workflow.module.main(workflow.csv)
    workflow.constructor.assert_not_called()
    workflow.execute.assert_not_called()


def test_execution_failure_is_preserved_and_prevents_second_request(workflow, capsys):
    original = ValueError("tool calculation failed")
    workflow.execute.side_effect = original
    with pytest.raises(ValueError) as caught:
        workflow.module.main(workflow.csv)
    assert caught.value is original
    assert workflow.create.call_count == 1
    assert "模型最终回答" not in capsys.readouterr().out


def test_nonfinite_tool_result_is_not_sent(workflow):
    workflow.execute.side_effect = None
    workflow.execute.return_value = {"invalid": float("inf")}
    with pytest.raises(ValueError):
        workflow.module.main(workflow.csv)
    assert workflow.create.call_count == 1


@pytest.mark.parametrize("round_number", [0, 1])
def test_api_errors_are_preserved_without_extra_requests(
    workflow, round_number, capsys
):
    original = OpenAIError("offline API failure")
    workflow.pending[round_number] = original
    with pytest.raises(OpenAIError) as caught:
        workflow.module.main(workflow.csv)
    assert caught.value is original
    assert workflow.create.call_count == round_number + 1
    assert workflow.execute.call_count == round_number
    assert "模型最终回答" not in capsys.readouterr().out


def test_no_second_choices(workflow):
    workflow.pending[1] = SimpleNamespace(choices=[])
    with pytest.raises(RuntimeError, match="第二轮.*choice"):
        workflow.module.main(workflow.csv)
    assert workflow.create.call_count == 2


@pytest.mark.parametrize("reason", ["length", None, "tool_calls"])
def test_abnormal_second_finish_is_not_presented_as_final_answer(
    workflow, reason, capsys
):
    workflow.pending[1] = response(content="partial", finish_reason=reason)
    with pytest.raises(RuntimeError, match="第二轮.*未正常结束"):
        workflow.module.main(workflow.csv)
    assert "模型最终回答" not in capsys.readouterr().out
    assert workflow.create.call_count == 2


def test_second_refusal_is_rejected(workflow):
    workflow.pending[1] = response(content="ignored", refusal="declined")
    with pytest.raises(RuntimeError, match="第二轮.*拒绝"):
        workflow.module.main(workflow.csv)


def test_unexpected_second_tool_call_is_rejected(workflow):
    workflow.pending[1] = response(
        calls=[tool_call(identifier="unexpected")], content="ignored"
    )
    with pytest.raises(RuntimeError, match="第二轮意外"):
        workflow.module.main(workflow.csv)
    assert workflow.create.call_count == 2
    workflow.execute.assert_called_once()


@pytest.mark.parametrize("content", [None, "", " \n\t"])
def test_blank_final_answer_is_rejected(workflow, content):
    workflow.pending[1] = response(content=content)
    with pytest.raises(RuntimeError, match="有效正文"):
        workflow.module.main(workflow.csv)
