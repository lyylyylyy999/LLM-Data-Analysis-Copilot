"""Offline validation of one tool execution and its two-request message chain."""

import copy
import csv
import importlib.util
import json
import socket
import statistics
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from openai import OpenAIError
from openai.types.chat import ChatCompletionMessageFunctionToolCall

from data_copilot.analysis_tools import dataset_info as real_dataset_info
from data_copilot.analysis_tools import describe_numeric as real_describe_numeric
from data_copilot.analysis_tools import group_aggregate as real_group_aggregate
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

    def describe(dataframe, columns):
        events.append("execute")
        return real_describe_numeric(dataframe, columns)

    describe_mock = Mock(side_effect=describe)

    def aggregate(dataframe, **kwargs):
        events.append("execute")
        return real_group_aggregate(dataframe, **kwargs)

    aggregate_mock = Mock(side_effect=aggregate)

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
    monkeypatch.setattr(module, "describe_numeric", describe_mock)
    monkeypatch.setattr(module, "group_aggregate", aggregate_mock)
    csv_path = tmp_path / "tiny.csv"
    csv_path.write_text("region,sales\nEast,10\nWest,\nNorth,30\n", encoding="utf-8")
    return SimpleNamespace(
        module=module,
        create=create,
        constructor=constructor,
        execute=execute_mock,
        describe=describe_mock,
        aggregate=aggregate_mock,
        read=read,
        pending=pending,
        requests=requests,
        events=events,
        dotenv=dotenv,
        csv=csv_path,
    )


def assert_no_execution_or_second_request(workflow):
    workflow.execute.assert_not_called()
    workflow.describe.assert_not_called()
    workflow.aggregate.assert_not_called()
    assert workflow.create.call_count == 1


def test_import_has_no_io(workflow):
    workflow.constructor.assert_not_called()
    workflow.create.assert_not_called()
    workflow.execute.assert_not_called()
    workflow.describe.assert_not_called()
    workflow.aggregate.assert_not_called()
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
    assert first["messages"][0]["role"] == "system"
    system = first["messages"][0]["content"]
    assert system.startswith(workflow.module.SYSTEM_PROMPT)
    metadata = json.loads(system[system.index("{") :])
    assert metadata == {name: str(dtype) for name, dtype in dataframe.dtypes.items()}
    assert first["messages"][1] == {"role": "user", "content": workflow.module.QUESTION}
    assert "East" not in system and "North" not in system
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
    with pytest.raises(ValueError, match="只支持一次工具调用"):
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


def test_three_tool_schemas_do_not_change_task08_tool_definitions(workflow):
    from examples.tool_calling import TOOLS as observation_tools

    functions = {
        tool["function"]["name"]: tool["function"] for tool in workflow.module.TOOLS
    }
    assert set(functions) == {"dataset_info", "describe_numeric", "group_aggregate"}
    assert [tool["function"]["name"] for tool in observation_tools] == ["dataset_info"]
    assert functions["dataset_info"]["parameters"] == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    parameters = functions["describe_numeric"]["parameters"]
    assert parameters["type"] == "object"
    assert parameters["required"] == ["columns"]
    assert parameters["additionalProperties"] is False
    assert set(parameters["properties"]) == {"columns"}
    columns = parameters["properties"]["columns"]
    assert columns["type"] == "array"
    assert columns["items"]["type"] == "string"
    assert columns["minItems"] == 1
    json.dumps(workflow.module.TOOLS, allow_nan=False)


@pytest.mark.parametrize("columns", [["sales"], ["sales", "advertising_spend"]])
def test_descriptive_tool_end_to_end_uses_model_columns(workflow, columns, capsys):
    workflow.csv.write_text(
        "region,sales,advertising_spend\nEast,10,1\nWest,,3\nNorth,30,5\n",
        encoding="utf-8",
    )
    original = real_load_csv(workflow.csv)
    arguments = json.dumps({"columns": columns})
    workflow.pending[0] = response(
        calls=[
            tool_call(
                name="describe_numeric", identifier="describe_1", arguments=arguments
            )
        ],
        finish_reason="tool_calls",
    )
    question = "请描述指定列的统计特征"
    workflow.pending[1] = response(content="以下为模型的模拟解释。")
    workflow.module.main(workflow.csv, question)
    workflow.execute.assert_not_called()
    workflow.describe.assert_called_once()
    frame, selected = workflow.describe.call_args.args
    assert selected == columns
    pd.testing.assert_frame_equal(frame, original)
    assert workflow.events == ["request", "execute", "request"]
    first, second = workflow.requests
    assert first["messages"][1] == {"role": "user", "content": question}
    system = first["messages"][0]["content"]
    metadata = json.loads(system[system.index("{") :])
    assert metadata == {name: str(dtype) for name, dtype in original.dtypes.items()}
    assert "East" not in system
    assert second["messages"][:2] == first["messages"]
    assistant, tool = second["messages"][2:]
    assert assistant["tool_calls"] == [
        {
            "id": "describe_1",
            "type": "function",
            "function": {"name": "describe_numeric", "arguments": arguments},
        }
    ]
    assert tool["role"] == "tool" and tool["tool_call_id"] == "describe_1"
    result = json.loads(tool["content"])
    assert result["ddof"] == 1
    assert set(result["statistics"]) == set(columns)
    sales = result["statistics"]["sales"]
    assert sales == {
        "count": 2,
        "mean": 20.0,
        "std": pytest.approx(200**0.5),
        "min": 10.0,
        "25%": 15.0,
        "median": 20.0,
        "75%": 25.0,
        "max": 30.0,
        "unavailable_statistics": [],
    }
    if "advertising_spend" in columns:
        assert result["statistics"]["advertising_spend"] == {
            "count": 3,
            "mean": 3.0,
            "std": 2.0,
            "min": 1.0,
            "25%": 2.0,
            "median": 3.0,
            "75%": 4.0,
            "max": 5.0,
            "unavailable_statistics": [],
        }
    assert "tools" not in second and "response_format" not in second
    json.dumps(result, allow_nan=False)
    assert "模型最终回答：以下为模型的模拟解释。" in capsys.readouterr().out


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"columns": "sales"},
        {"columns": []},
        {"columns": [1]},
        {"columns": [True]},
        {"columns": [["sales"]]},
        {"columns": ["sales"], "ddof": 0},
    ],
)
def test_bad_describe_argument_shape_stops_before_calculation(workflow, value):
    workflow.pending[0] = response(
        calls=[tool_call(name="describe_numeric", arguments=json.dumps(value))],
        finish_reason="tool_calls",
    )
    with pytest.raises(ValueError):
        workflow.module.main(workflow.csv)
    assert_no_execution_or_second_request(workflow)


@pytest.mark.parametrize(
    "columns", [["sales", "sales"], ["absent"], ["region"], [""], [" "]]
)
def test_invalid_selected_columns_use_existing_numeric_validation(workflow, columns):
    workflow.pending[0] = response(
        calls=[
            tool_call(
                name="describe_numeric", arguments=json.dumps({"columns": columns})
            )
        ],
        finish_reason="tool_calls",
    )
    with pytest.raises(ValueError):
        workflow.module.main(workflow.csv)
    workflow.execute.assert_not_called()
    workflow.describe.assert_called_once()
    assert workflow.create.call_count == 1


@pytest.mark.parametrize("arguments", ["not JSON", "[]", "null"])
def test_describe_requires_json_object(workflow, arguments):
    workflow.pending[0] = response(
        calls=[tool_call(name="describe_numeric", arguments=arguments)],
        finish_reason="tool_calls",
    )
    expected = json.JSONDecodeError if arguments == "not JSON" else ValueError
    with pytest.raises(expected):
        workflow.module.main(workflow.csv)
    assert_no_execution_or_second_request(workflow)


def test_dataset_info_does_not_accept_columns(workflow):
    frame = real_load_csv(workflow.csv)
    with pytest.raises(ValueError, match="不接受参数"):
        workflow.module.execute_tool(frame, "dataset_info", '{"columns":["sales"]}')
    workflow.execute.assert_not_called()
    workflow.describe.assert_not_called()


def test_model_cannot_request_unregistered_tool_with_valid_arguments(workflow):
    frame = real_load_csv(workflow.csv)
    with pytest.raises(ValueError, match="工具名称"):
        workflow.module.execute_tool(frame, "correlation", "{}")
    workflow.execute.assert_not_called()
    workflow.describe.assert_not_called()


def test_all_missing_description_is_serialized_as_null(workflow):
    workflow.csv.write_text("sales,region\n,A\n,B\n", encoding="utf-8")
    workflow.pending[0] = response(
        calls=[tool_call(name="describe_numeric", arguments='{"columns":["sales"]}')],
        finish_reason="tool_calls",
    )
    workflow.module.main(workflow.csv)
    result = json.loads(workflow.requests[1]["messages"][-1]["content"])
    stats = result["statistics"]["sales"]
    assert result["ddof"] == 1
    assert stats["count"] == 0
    for field in ["mean", "std", "min", "25%", "median", "75%", "max"]:
        assert stats[field] is None
        assert field in stats["unavailable_statistics"]
    json.dumps(result, allow_nan=False)


def test_descriptive_calculation_failure_prevents_second_round(workflow, capsys):
    original = RuntimeError("descriptive calculation failed")
    workflow.describe.side_effect = original
    workflow.pending[0] = response(
        calls=[tool_call(name="describe_numeric", arguments='{"columns":["sales"]}')],
        finish_reason="tool_calls",
    )
    with pytest.raises(RuntimeError) as caught:
        workflow.module.main(workflow.csv)
    assert caught.value is original
    assert workflow.create.call_count == 1
    workflow.execute.assert_not_called()
    assert "模型最终回答" not in capsys.readouterr().out


@pytest.mark.parametrize("question", ["", " \n\t"])
def test_blank_question_prevents_reading_and_requesting(workflow, question):
    with pytest.raises(ValueError, match="问题不能为空"):
        workflow.module.main(workflow.csv, question)
    workflow.read.assert_not_called()
    workflow.create.assert_not_called()


def test_python_function_accepts_string_but_tool_contract_requires_list(workflow):
    frame = real_load_csv(workflow.csv)
    assert real_describe_numeric(frame, "sales")["statistics"]["sales"]["mean"] == 20.0
    with pytest.raises(ValueError, match="字符串数组"):
        workflow.module.execute_tool(frame, "describe_numeric", '{"columns":"sales"}')
    workflow.describe.assert_not_called()


def test_group_schema_has_exact_required_arguments_and_enum(workflow):
    tools = {
        tool["function"]["name"]: tool["function"] for tool in workflow.module.TOOLS
    }
    parameters = tools["group_aggregate"]["parameters"]
    names = {"group_column", "target_column", "aggregation"}
    assert parameters["type"] == "object"
    assert set(parameters["properties"]) == names
    assert set(parameters["required"]) == names
    assert parameters["additionalProperties"] is False
    for name in names:
        assert parameters["properties"][name]["type"] == "string"
    assert set(parameters["properties"]["aggregation"]["enum"]) == {
        "mean",
        "sum",
        "count",
        "min",
        "max",
    }


@pytest.mark.parametrize(
    "aggregation,values",
    [
        ("mean", [15.0, 30.0, None, 0.0]),
        ("sum", [30.0, 30.0, None, 0.0]),
        ("count", [2, 1, 0, 1]),
        ("min", [10.0, 30.0, None, 0.0]),
        ("max", [20.0, 30.0, None, 0.0]),
    ],
)
def test_group_end_to_end_preserves_missing_keys_and_values(
    workflow, aggregation, values, capsys
):
    workflow.csv.write_text(
        "region,sales,category\nA,10,X\nA,20,Y\nA,,X\n,30,X\nB,,Y\nC,0,X\n",
        encoding="utf-8",
    )
    original = real_load_csv(workflow.csv)
    arguments = json.dumps(
        {
            "group_column": "region",
            "target_column": "sales",
            "aggregation": aggregation,
        }
    )
    workflow.pending[0] = response(
        calls=[
            tool_call(
                name="group_aggregate",
                identifier="group_call",
                arguments=arguments,
            )
        ],
        finish_reason="tool_calls",
    )
    workflow.pending[1] = response(content="模拟解释，不用于评估模型回答质量。")
    question = "请按地区汇总销售额"
    workflow.module.main(workflow.csv, question)
    workflow.execute.assert_not_called()
    workflow.describe.assert_not_called()
    workflow.aggregate.assert_called_once()
    frame = workflow.aggregate.call_args.args[0]
    pd.testing.assert_frame_equal(frame, original)
    assert workflow.aggregate.call_args.kwargs == {
        "group_column": "region",
        "target_column": "sales",
        "aggregation": aggregation,
    }
    assert workflow.events == ["request", "execute", "request"]
    first, second = workflow.requests
    assert first["messages"][1] == {"role": "user", "content": question}
    system = first["messages"][0]["content"]
    metadata = json.loads(system[system.index("{") :])
    assert metadata == {name: str(dtype) for name, dtype in original.dtypes.items()}
    assert second["messages"][:2] == first["messages"]
    assistant, tool = second["messages"][2:]
    assert assistant["tool_calls"] == [
        {
            "id": "group_call",
            "type": "function",
            "function": {"name": "group_aggregate", "arguments": arguments},
        }
    ]
    assert tool["role"] == "tool" and tool["tool_call_id"] == "group_call"
    result = json.loads(tool["content"])
    assert result == {
        "group_column": "region",
        "target_column": "sales",
        "aggregation": aggregation,
        "groups": [
            {
                "group": key,
                "value": value,
                "status": "ok" if value is not None else "undefined_or_nonfinite",
            }
            for key, value in zip(["A", None, "B", "C"], values, strict=True)
        ],
    }
    # B has one missing target; C has a genuine zero. These must stay distinct.
    assert result["groups"][2]["value"] == (0 if aggregation == "count" else None)
    assert result["groups"][3]["value"] == (1 if aggregation == "count" else 0)
    assert "tools" not in second and "response_format" not in second
    json.dumps(result, allow_nan=False)
    assert "模型最终回答：模拟解释" in capsys.readouterr().out


@pytest.mark.parametrize(
    "group_column,aggregation",
    [
        ("region", "mean"),
        ("category", "sum"),
        ("region", "count"),
    ],
)
def test_real_sales_group_results_match_independent_csv_calculation(
    workflow, group_column, aggregation
):
    path = Path(__file__).resolve().parents[1] / "data/samples/sales.csv"
    arguments = json.dumps(
        {
            "group_column": group_column,
            "target_column": "sales",
            "aggregation": aggregation,
        }
    )
    workflow.pending[0] = response(
        calls=[tool_call(name="group_aggregate", arguments=arguments)],
        finish_reason="tool_calls",
    )
    workflow.module.main(path)
    result = json.loads(workflow.requests[1]["messages"][-1]["content"])
    grouped = {}
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            values = grouped.setdefault(row[group_column] or None, [])
            if row["sales"]:
                values.append(float(row["sales"]))
    assert [item["group"] for item in result["groups"]] == list(grouped)
    for item in result["groups"]:
        values = grouped[item["group"]]
        expected = (
            len(values)
            if aggregation == "count"
            else sum(values)
            if aggregation == "sum"
            else statistics.mean(values)
        )
        assert item["value"] == pytest.approx(expected)
        assert item["status"] == "ok"


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"group_column": "region", "target_column": "sales"},
        {"group_column": "region", "aggregation": "mean"},
        {"target_column": "sales", "aggregation": "mean"},
        {
            "group_column": "region",
            "target_column": "sales",
            "aggregation": "mean",
            "ddof": 0,
        },
    ],
)
def test_missing_or_extra_group_fields_prevent_calculation(workflow, arguments):
    workflow.pending[0] = response(
        calls=[
            tool_call(
                name="group_aggregate",
                arguments=json.dumps(arguments),
            )
        ],
        finish_reason="tool_calls",
    )
    with pytest.raises(ValueError, match="三个必填参数"):
        workflow.module.main(workflow.csv)
    assert_no_execution_or_second_request(workflow)


@pytest.mark.parametrize("field", ["group_column", "target_column", "aggregation"])
@pytest.mark.parametrize("value", [None, 1, True, []])
def test_group_argument_types_are_strict(workflow, field, value):
    arguments = {
        "group_column": "region",
        "target_column": "sales",
        "aggregation": "mean",
    }
    arguments[field] = value
    workflow.pending[0] = response(
        calls=[
            tool_call(
                name="group_aggregate",
                arguments=json.dumps(arguments),
            )
        ],
        finish_reason="tool_calls",
    )
    with pytest.raises(ValueError, match="必须是字符串"):
        workflow.module.main(workflow.csv)
    assert_no_execution_or_second_request(workflow)


@pytest.mark.parametrize(
    "field,value",
    [
        ("group_column", "unknown"),
        ("target_column", "unknown"),
        ("group_column", ""),
        ("target_column", " "),
        ("target_column", "region"),
        ("aggregation", "median"),
        ("aggregation", "MEAN"),
        ("aggregation", ""),
    ],
)
def test_group_semantics_are_checked_by_real_tool(workflow, field, value):
    arguments = {
        "group_column": "region",
        "target_column": "sales",
        "aggregation": "mean",
    }
    arguments[field] = value
    workflow.pending[0] = response(
        calls=[
            tool_call(
                name="group_aggregate",
                arguments=json.dumps(arguments),
            )
        ],
        finish_reason="tool_calls",
    )
    with pytest.raises(ValueError):
        workflow.module.main(workflow.csv)
    workflow.execute.assert_not_called()
    workflow.describe.assert_not_called()
    workflow.aggregate.assert_called_once()
    assert workflow.create.call_count == 1


@pytest.mark.parametrize("arguments", ["not JSON", "[]", "null"])
def test_group_arguments_must_be_json_object(workflow, arguments):
    workflow.pending[0] = response(
        calls=[
            tool_call(
                name="group_aggregate",
                arguments=arguments,
            )
        ],
        finish_reason="tool_calls",
    )
    expected = json.JSONDecodeError if arguments == "not JSON" else ValueError
    with pytest.raises(expected):
        workflow.module.main(workflow.csv)
    assert_no_execution_or_second_request(workflow)


def test_group_failure_is_preserved_without_second_request(workflow, capsys):
    original = RuntimeError("group calculation failed")
    workflow.aggregate.side_effect = original
    arguments = '{"group_column":"region","target_column":"sales","aggregation":"mean"}'
    workflow.pending[0] = response(
        calls=[
            tool_call(
                name="group_aggregate",
                arguments=arguments,
            )
        ],
        finish_reason="tool_calls",
    )
    with pytest.raises(RuntimeError) as caught:
        workflow.module.main(workflow.csv)
    assert caught.value is original
    assert workflow.create.call_count == 1
    assert "模型最终回答" not in capsys.readouterr().out
