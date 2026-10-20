"""Offline validation of tool batches and bounded model request histories."""

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

from data_copilot.analysis_tools import correlation as real_correlation
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

    def correlate(dataframe, **kwargs):
        events.append("execute")
        return real_correlation(dataframe, **kwargs)

    correlate_mock = Mock(side_effect=correlate)

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
    from data_copilot import tool_registry as registry

    service_path = (
        Path(__file__).resolve().parents[1] / "src/data_copilot/analysis_service.py"
    )
    service_spec = importlib.util.spec_from_file_location(
        "analysis_service_test", service_path
    )
    assert service_spec is not None and service_spec.loader is not None
    service = importlib.util.module_from_spec(service_spec)
    service_spec.loader.exec_module(service)
    path = Path(__file__).resolve().parents[1] / "examples/tool_execution.py"
    spec = importlib.util.spec_from_file_location("tool_execution_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    read = Mock(wraps=real_load_csv)
    monkeypatch.setattr(module, "load_csv", read)
    monkeypatch.setattr(module, "analyze_question", service.analyze_question)
    monkeypatch.setattr(registry, "dataset_info", execute_mock)
    monkeypatch.setattr(registry, "describe_numeric", describe_mock)
    monkeypatch.setattr(registry, "group_aggregate", aggregate_mock)
    monkeypatch.setattr(registry, "correlation", correlate_mock)
    csv_path = tmp_path / "tiny.csv"
    csv_path.write_text("region,sales\nEast,10\nWest,\nNorth,30\n", encoding="utf-8")
    return SimpleNamespace(
        module=module,
        service=service,
        registry=registry,
        create=create,
        constructor=constructor,
        execute=execute_mock,
        describe=describe_mock,
        aggregate=aggregate_mock,
        correlate=correlate_mock,
        read=read,
        pending=pending,
        requests=requests,
        events=events,
        dotenv=dotenv,
        csv=csv_path,
    )


def test_service_returns_complete_payload_without_printing(workflow, capsys):
    frame = pd.DataFrame({"sales": [10, None, 30]})
    before = frame.copy(deep=True)
    workflow.pending[1].model = "actual-final-model"
    result = workflow.service.analyze_question(frame, "多少行？")
    assert result == {
        "answer": "数据有3行、2列。",  # Mocked text is not a model accuracy assertion.
        "model": "actual-final-model",
        "finish_reason": "stop",
        "tool_results": [
            {
                "tool_call_id": "call_1",
                "name": "dataset_info",
                "arguments": {},
                "result": real_dataset_info(before),
            }
        ],
    }
    assert workflow.execute.call_args.args[0] is frame
    workflow.read.assert_not_called()
    pd.testing.assert_frame_equal(frame, before)
    assert capsys.readouterr() == ("", "")
    json.dumps(result, ensure_ascii=False, allow_nan=False)


def test_service_returns_all_batch_results_in_order(workflow, capsys):
    frame = pd.DataFrame({"sales": [10, 20, 30]})
    calls = [
        tool_call(identifier="a", arguments="{ }"),
        tool_call(
            identifier="b", name="describe_numeric", arguments='{"columns":["sales"]}'
        ),
    ]
    workflow.pending[0] = response(calls=calls, finish_reason="tool_calls")
    result = workflow.service.analyze_question(frame, "规模和销售统计？")
    expected = [real_dataset_info(frame), real_describe_numeric(frame, ["sales"])]
    assert result["tool_results"] == [
        {
            "tool_call_id": call.id,
            "name": call.function.name,
            "arguments": json.loads(call.function.arguments),
            "result": value,
        }
        for call, value in zip(calls, expected, strict=True)
    ]
    for item, message in zip(
        result["tool_results"], workflow.requests[1]["messages"][3:], strict=True
    ):
        assert item["tool_call_id"] == message["tool_call_id"]
        assert item["result"] == json.loads(message["content"])
    assert capsys.readouterr() == ("", "")
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("calls", [None, []])
def test_service_direct_answer_returns_first_model_and_no_tools(
    workflow, calls, capsys
):
    workflow.pending[0] = response(calls=calls, content=" 相关不代表因果。 ")
    workflow.pending[0].model = "actual-first-model"
    result = workflow.service.analyze_question(pd.DataFrame(), "相关代表因果吗？")
    assert result == {
        "answer": " 相关不代表因果。 ",
        "model": "actual-first-model",
        "finish_reason": "stop",
        "tool_results": [],
    }
    assert_no_execution_or_second_request(workflow)
    assert capsys.readouterr() == ("", "")
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("content", [None, "", " \n\t", 42])
def test_service_rejects_invalid_direct_answer(workflow, content):
    workflow.pending[0] = response(content=content)
    with pytest.raises(RuntimeError, match="第1次请求.*正文"):
        workflow.service.analyze_question(pd.DataFrame(), "问题")
    assert_no_execution_or_second_request(workflow)


@pytest.mark.parametrize("question", [None, 1, "", " \n\t"])
def test_service_rejects_invalid_question_before_config_or_client(workflow, question):
    with pytest.raises(ValueError):
        workflow.service.analyze_question(pd.DataFrame(), question)
    workflow.dotenv.assert_not_called()
    workflow.constructor.assert_not_called()


@pytest.mark.parametrize("frame", [None, {}, [], "data.csv"])
def test_service_rejects_non_dataframe_before_config_or_client(workflow, frame):
    with pytest.raises(TypeError, match="DataFrame"):
        workflow.service.analyze_question(frame, "问题")
    workflow.dotenv.assert_not_called()
    workflow.constructor.assert_not_called()


def test_service_config_is_read_per_call_and_calls_do_not_share_history(
    workflow, monkeypatch
):
    workflow.pending[:] = [response(content="回答一"), response(content="回答二")]
    frame = pd.DataFrame({"sales": [1]})
    workflow.service.analyze_question(frame, "问题一")
    monkeypatch.setenv("DASHSCOPE_MODEL", "new-configured-model")
    workflow.service.analyze_question(frame, "问题二")
    assert workflow.dotenv.call_count == workflow.constructor.call_count == 2
    assert [request["model"] for request in workflow.requests] == [
        "configured-model",
        "new-configured-model",
    ]
    for request, question in zip(workflow.requests, ["问题一", "问题二"], strict=True):
        assert len(request["messages"]) == 2
        assert request["messages"][1] == {"role": "user", "content": question}


def test_cli_only_loads_delegates_and_displays(workflow, monkeypatch, capsys):
    payload = {
        "answer": "示例回答",
        "model": "final-model",
        "finish_reason": "stop",
        "tool_results": [],
    }
    analyze = Mock(return_value=payload)
    monkeypatch.setattr(workflow.module, "analyze_question", analyze)
    result = workflow.module.main(workflow.csv, "一个问题")
    assert result is payload
    workflow.read.assert_called_once_with(workflow.csv)
    analyze.assert_called_once()
    pd.testing.assert_frame_equal(
        analyze.call_args.args[0], real_load_csv(workflow.csv)
    )
    assert analyze.call_args.args[1] == "一个问题"
    assert "模型最终回答：示例回答" in capsys.readouterr().out
    workflow.constructor.assert_not_called()


def test_new_modules_import_without_config_client_or_output(
    workflow, monkeypatch, capsys
):
    forbidden = Mock(
        side_effect=AssertionError("Import must not read config or create client")
    )
    monkeypatch.setattr("os.getenv", forbidden)
    monkeypatch.setattr("dotenv.load_dotenv", forbidden)
    monkeypatch.setattr("openai.OpenAI", forbidden)
    root = Path(__file__).resolve().parents[1] / "src/data_copilot"
    for name in ("tool_registry", "analysis_service"):
        spec = importlib.util.spec_from_file_location(
            f"import_check_{name}", root / f"{name}.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    forbidden.assert_not_called()
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("limit", [True, False, 0, -1, None, 1.0, "2", [], {}])
def test_invalid_round_limit_is_rejected_before_config(workflow, limit):
    with pytest.raises(ValueError, match="max_tool_rounds.*正整数"):
        workflow.service.analyze_question(pd.DataFrame(), "问题", max_tool_rounds=limit)
    workflow.dotenv.assert_not_called()
    workflow.constructor.assert_not_called()
    workflow.create.assert_not_called()


def test_dependent_tool_rounds_preserve_history_results_and_final_model(
    workflow, capsys
):
    frame = pd.DataFrame({"sales": [10, None, 30], "quantity": [1, 2, 3]})
    before = frame.copy(deep=True)
    first_call = tool_call(identifier="info", arguments="{ }")
    second_call = tool_call(
        identifier="stats", name="describe_numeric", arguments='{"columns":["sales"]}'
    )
    workflow.pending[:] = [
        response(calls=[first_call], content="先检查缺失", finish_reason="tool_calls"),
        response(
            calls=[second_call], content="再分析缺失列", finish_reason="tool_calls"
        ),
        response(content="模拟最终说明"),
    ]
    workflow.pending[-1].model = "third-response-model"
    result = workflow.service.analyze_question(
        frame, "先检查缺失，再分析", max_tool_rounds=2
    )
    assert workflow.events == ["request", "execute", "request", "execute", "request"]
    expected = [real_dataset_info(before), real_describe_numeric(before, ["sales"])]
    assert result == {
        "answer": "模拟最终说明",
        "model": "third-response-model",
        "finish_reason": "stop",
        "tool_results": [
            {
                "tool_call_id": call.id,
                "name": call.function.name,
                "arguments": json.loads(call.function.arguments),
                "result": value,
            }
            for call, value in zip([first_call, second_call], expected, strict=True)
        ],
    }
    for index, request in enumerate(workflow.requests):
        assert request["tool_choice"] == "auto"
        assert request["tools"] == workflow.registry.TOOLS
        assert request["parallel_tool_calls"] is True
        assert len(request["messages"]) == 2 + index * 2
        if index:
            previous = workflow.requests[index - 1]["messages"]
            assert request["messages"][:-2] == previous
            assistant, tool_message = request["messages"][-2:]
            call = [first_call, second_call][index - 1]
            assert assistant == {
                "role": "assistant",
                "content": ["先检查缺失", "再分析缺失列"][index - 1],
                "tool_calls": [call.model_dump()],
            }
            assert tool_message == {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(
                    expected[index - 1], ensure_ascii=False, allow_nan=False
                ),
            }
    pd.testing.assert_frame_equal(frame, before)
    workflow.read.assert_not_called()
    assert capsys.readouterr() == ("", "")
    json.dumps(result, ensure_ascii=False, allow_nan=False)


@pytest.mark.parametrize("limit", [1, 2, 3])
@pytest.mark.parametrize("answer_at_limit", [True, False])
def test_round_budget_counts_batches_not_calls(workflow, limit, answer_at_limit):
    # Two calls in every batch: reaching the limit depends on batches, not call count.
    workflow.pending[:] = [
        response(
            calls=[tool_call(identifier=f"r{r}_a"), tool_call(identifier=f"r{r}_b")],
            finish_reason="tool_calls",
        )
        for r in range(limit)
    ]
    final = (
        response(content="已完成")
        if answer_at_limit
        else response(
            calls=[tool_call(identifier="over_budget")], finish_reason="tool_calls"
        )
    )
    workflow.pending.append(final)
    if answer_at_limit:
        result = workflow.service.analyze_question(
            pd.DataFrame({"x": [1]}), "问题", max_tool_rounds=limit
        )
        assert len(result["tool_results"]) == limit * 2
        assert [item["tool_call_id"] for item in result["tool_results"]] == [
            f"r{r}_{suffix}" for r in range(limit) for suffix in ("a", "b")
        ]
    else:
        with pytest.raises(RuntimeError, match=f"轮数.*上限 {limit}"):
            workflow.service.analyze_question(
                pd.DataFrame({"x": [1]}), "问题", max_tool_rounds=limit
            )
    assert workflow.create.call_count == limit + 1
    assert workflow.execute.call_count == limit * 2
    assert len(workflow.requests[-1]["messages"]) == 2 + 3 * limit
    assert workflow.pending == []


def test_default_limit_stops_after_three_batches(workflow):
    workflow.pending[:] = [
        response(calls=[tool_call(identifier=f"r{i}")], finish_reason="tool_calls")
        for i in range(4)
    ]
    with pytest.raises(RuntimeError, match="轮数.*上限 3"):
        workflow.service.analyze_question(pd.DataFrame(), "问题")
    assert workflow.execute.call_count == 3
    assert workflow.create.call_count == 4


@pytest.mark.parametrize("limit", [1, 5])
def test_direct_answer_needs_only_one_request_regardless_of_budget(workflow, limit):
    workflow.pending[:] = [response(content="无需工具的回答")]
    result = workflow.service.analyze_question(
        pd.DataFrame(), "问题", max_tool_rounds=limit
    )
    assert result["tool_results"] == []
    assert_no_execution_or_second_request(workflow)


@pytest.mark.parametrize(
    "failure,pattern",
    [
        (SimpleNamespace(choices=[]), "choice"),
        (response(content="截断正文", finish_reason="length"), "截断"),
        (response(refusal="拒绝"), "拒绝"),
        (response(content="受限", finish_reason="content_filter"), "状态异常"),
        (response(calls=[], finish_reason="tool_calls"), "未返回工具调用"),
        (response(calls=[tool_call()], finish_reason="stop"), "结束原因不是"),
        (response(content=" \n"), "有效正文"),
    ],
)
def test_third_response_validation_stops_without_more_execution(
    workflow, failure, pattern
):
    workflow.pending[:] = [
        response(calls=[tool_call(identifier=f"r{i}")], finish_reason="tool_calls")
        for i in range(2)
    ] + [failure]
    with pytest.raises(RuntimeError, match=f"第3次请求.*{pattern}"):
        workflow.service.analyze_question(pd.DataFrame(), "问题")
    assert workflow.create.call_count == 3
    assert workflow.execute.call_count == 2


def test_later_api_error_preserves_original_exception_and_stops(workflow):
    original = OpenAIError("third request failed")
    workflow.pending[:] = [
        response(calls=[tool_call(identifier=f"r{i}")], finish_reason="tool_calls")
        for i in range(2)
    ] + [original]
    with pytest.raises(OpenAIError) as caught:
        workflow.service.analyze_question(pd.DataFrame(), "问题")
    assert caught.value is original
    assert workflow.create.call_count == 3
    assert workflow.execute.call_count == 2


@pytest.mark.parametrize(
    "failure", ["id", "type", "duplicate", "json", "calculation", "serialize"]
)
def test_later_tool_batch_failure_stops_without_next_request(workflow, failure):
    second = [tool_call(identifier="second")]
    original = ValueError("second batch failed")
    if failure == "id":
        second = [SimpleNamespace(type="function", id=" ")]
    elif failure == "type":
        second = [SimpleNamespace(type="custom")]
    elif failure == "duplicate":
        second *= 2
    elif failure == "json":
        second[0].function.arguments = "invalid JSON"
    elif failure == "calculation":
        workflow.execute.side_effect = [real_dataset_info(pd.DataFrame()), original]
    elif failure == "serialize":
        workflow.execute.side_effect = [
            real_dataset_info(pd.DataFrame()),
            {"x": float("inf")},
        ]
    workflow.pending[:] = [
        response(calls=[tool_call()], finish_reason="tool_calls"),
        response(calls=second, finish_reason="tool_calls"),
    ]
    with pytest.raises(ValueError) as caught:
        workflow.service.analyze_question(pd.DataFrame(), "问题")
    if failure == "calculation":
        assert caught.value is original
    assert workflow.create.call_count == 2
    expected_executions = 2 if failure in {"calculation", "serialize"} else 1
    assert workflow.execute.call_count == expected_executions


def assert_no_execution_or_second_request(workflow):
    workflow.execute.assert_not_called()
    workflow.describe.assert_not_called()
    workflow.aggregate.assert_not_called()
    workflow.correlate.assert_not_called()
    assert workflow.create.call_count == 1


def assert_batch_message_chain(workflow, calls, expected_results):
    first, second = workflow.requests
    assert first["parallel_tool_calls"] is True
    assert second["tool_choice"] == "auto"
    assert second["messages"][:2] == first["messages"]
    assistant, *results = second["messages"][2:]
    assert assistant == {
        "role": "assistant",
        "content": "准备执行整批分析",
        "tool_calls": [call.model_dump() for call in calls],
    }
    assert len(results) == len(calls) == len(expected_results)
    for call, message, expected in zip(calls, results, expected_results, strict=True):
        assert message == {
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(expected, ensure_ascii=False, allow_nan=False),
        }
    assert workflow.events == ["request", *["execute"] * len(calls), "request"]
    assert workflow.create.call_count == 2
    json.dumps(second, allow_nan=False)


def test_batch_of_different_tools_preserves_messages_and_results(workflow, capsys):
    workflow.csv.write_text("region,sales,ads\nEast,10,1\nWest,20,2\nEast,30,3\n")
    calls = [
        tool_call(identifier="info", arguments="{ }"),
        tool_call(
            identifier="describe",
            name="describe_numeric",
            arguments='{"columns": ["sales"]}',
        ),
        tool_call(
            identifier="corr",
            name="correlation",
            arguments='{"x_column":"sales","y_column":"ads","method":"pearson"}',
        ),
    ]
    workflow.pending[0] = response(
        calls=calls, content="准备执行整批分析", finish_reason="tool_calls"
    )
    before = real_load_csv(workflow.csv)
    workflow.module.main(workflow.csv, "数据规模、销售额统计及广告相关性？")
    expected = [
        real_dataset_info(before),
        real_describe_numeric(before, ["sales"]),
        real_correlation(before, "sales", "ads"),
    ]
    assert_batch_message_chain(workflow, calls, expected)
    assert expected[0]["row_count"] == 3
    assert expected[1]["statistics"]["sales"]["mean"] == 20
    assert expected[2]["coefficient"] == pytest.approx(1)
    workflow.aggregate.assert_not_called()
    for mock in (workflow.execute, workflow.describe, workflow.correlate):
        mock.assert_called_once()
        pd.testing.assert_frame_equal(mock.call_args.args[0], before)
    output = capsys.readouterr().out
    for call in calls:
        assert f"ID={call.id}" in output
        assert f"名称={call.function.name}" in output


@pytest.mark.parametrize("same_arguments", [False, True])
def test_same_tool_multiple_times_is_not_deduplicated(workflow, same_arguments):
    workflow.csv.write_text("region,category,sales\nEast,A,10\nWest,A,20\nEast,B,30\n")
    arguments = [
        {"group_column": "region", "target_column": "sales", "aggregation": "mean"},
        {"group_column": "category", "target_column": "sales", "aggregation": "sum"},
    ]
    if same_arguments:
        arguments[1] = arguments[0].copy()
    calls = [
        tool_call(
            name="group_aggregate", identifier=f"group_{i}", arguments=json.dumps(a)
        )
        for i, a in enumerate(arguments)
    ]
    workflow.pending[0] = response(
        calls=calls, content="准备执行整批分析", finish_reason="tool_calls"
    )
    workflow.module.main(workflow.csv)
    before = real_load_csv(workflow.csv)
    expected = [real_group_aggregate(before, **a) for a in arguments]
    assert_batch_message_chain(workflow, calls, expected)
    assert workflow.aggregate.call_count == 2
    for actual, requested in zip(
        workflow.aggregate.call_args_list, arguments, strict=True
    ):
        assert actual.kwargs == requested
        pd.testing.assert_frame_equal(actual.args[0], before)


@pytest.mark.parametrize("identifier", [None, "", " \t", 123, True])
def test_later_invalid_id_prevents_any_batch_execution(workflow, identifier):
    bad = SimpleNamespace(type="function", id=identifier)
    workflow.pending[0] = response(calls=[tool_call(), bad], finish_reason="tool_calls")
    with pytest.raises(ValueError, match="ID"):
        workflow.module.main(workflow.csv)
    assert_no_execution_or_second_request(workflow)


def test_later_invalid_type_prevents_any_batch_execution(workflow):
    workflow.pending[0] = response(
        calls=[tool_call(), SimpleNamespace(type="custom")], finish_reason="tool_calls"
    )
    with pytest.raises(ValueError, match="调用类型"):
        workflow.module.main(workflow.csv)
    assert_no_execution_or_second_request(workflow)


def test_nonadjacent_duplicate_id_prevents_any_batch_execution(workflow):
    workflow.pending[0] = response(
        calls=[tool_call(), tool_call(identifier="other"), tool_call()],
        finish_reason="tool_calls",
    )
    with pytest.raises(ValueError, match="ID.*重复"):
        workflow.module.main(workflow.csv)
    assert_no_execution_or_second_request(workflow)


@pytest.mark.parametrize("failure_index", [0, 1, 2])
@pytest.mark.parametrize(
    "failure_kind", ["json", "unknown", "columns", "calculation", "serialize"]
)
def test_batch_failure_stops_remaining_execution_and_second_request(
    workflow, monkeypatch, failure_index, failure_kind, capsys
):
    calls = [tool_call(identifier=f"call_{i}") for i in range(3)]
    original = None
    if failure_kind == "json":
        calls[failure_index].function.arguments = "invalid JSON"
        expected_error = json.JSONDecodeError
    elif failure_kind == "unknown":
        calls[failure_index].function.name = "unknown_tool"
        expected_error = ValueError
    elif failure_kind == "columns":
        calls[failure_index].function.name = "describe_numeric"
        calls[failure_index].function.arguments = '{"columns":["absent"]}'
        expected_error = ValueError
    else:
        expected_error = ValueError
        original = ValueError("batch calculation failed")
    workflow.pending[0] = response(calls=calls, finish_reason="tool_calls")
    actual_execute = workflow.service.execute_tool
    attempted = []

    def tracking_execute(dataframe, name, arguments):
        index = len(attempted)
        attempted.append(name)
        if index == failure_index and failure_kind == "calculation":
            raise original
        result = actual_execute(dataframe, name, arguments)
        if index == failure_index and failure_kind == "serialize":
            return {"invalid": float("nan")}
        return result

    monkeypatch.setattr(workflow.service, "execute_tool", tracking_execute)
    with pytest.raises(expected_error) as caught:
        workflow.module.main(workflow.csv)
    if failure_kind == "calculation":
        assert caught.value is original
    assert len(attempted) == failure_index + 1
    assert workflow.create.call_count == 1
    assert len(workflow.requests[0]["messages"]) == 2
    assert "模型最终回答" not in capsys.readouterr().out


def test_import_has_no_io(workflow):
    workflow.constructor.assert_not_called()
    workflow.create.assert_not_called()
    workflow.execute.assert_not_called()
    workflow.describe.assert_not_called()
    workflow.aggregate.assert_not_called()
    workflow.correlate.assert_not_called()
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
    assert first["tools"] == workflow.registry.TOOLS
    assert len(first["messages"]) == 2
    assert first["messages"][0]["role"] == "system"
    system = first["messages"][0]["content"]
    assert system.startswith(workflow.service.SYSTEM_PROMPT)
    metadata = json.loads(system[system.index("{") :])
    assert metadata == {name: str(dtype) for name, dtype in dataframe.dtypes.items()}
    assert first["messages"][1] == {"role": "user", "content": workflow.module.QUESTION}
    assert "East" not in system and "North" not in system
    assert second["tools"] == workflow.registry.TOOLS
    assert second["tool_choice"] == "auto"
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
    with pytest.raises(RuntimeError, match="第1次请求.*choice"):
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
    with pytest.raises(RuntimeError, match="第1次请求.*拒绝"):
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


def test_duplicate_ids_prevent_entire_batch_execution(workflow):
    workflow.pending[0] = response(
        calls=[tool_call(), tool_call()], finish_reason="tool_calls"
    )
    with pytest.raises(ValueError, match="ID.*重复"):
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
    with pytest.raises(RuntimeError, match="第2次请求.*choice"):
        workflow.module.main(workflow.csv)
    assert workflow.create.call_count == 2


@pytest.mark.parametrize(
    "reason,detail",
    [("length", "截断"), (None, "状态异常"), ("tool_calls", "未返回工具调用")],
)
def test_abnormal_second_finish_is_not_presented_as_final_answer(
    workflow, reason, detail, capsys
):
    workflow.pending[1] = response(content="partial", finish_reason=reason)
    with pytest.raises(RuntimeError, match=f"第2次请求.*{detail}"):
        workflow.module.main(workflow.csv)
    assert "模型最终回答" not in capsys.readouterr().out
    assert workflow.create.call_count == 2


def test_second_refusal_is_rejected(workflow):
    workflow.pending[1] = response(content="ignored", refusal="declined")
    with pytest.raises(RuntimeError, match="第2次请求.*拒绝"):
        workflow.module.main(workflow.csv)


def test_unexpected_second_tool_call_is_rejected(workflow):
    workflow.pending[1] = response(
        calls=[tool_call(identifier="unexpected")], content="ignored"
    )
    with pytest.raises(RuntimeError, match="第2次请求意外"):
        workflow.module.main(workflow.csv)
    assert workflow.create.call_count == 2
    workflow.execute.assert_called_once()


@pytest.mark.parametrize("content", [None, "", " \n\t"])
def test_blank_final_answer_is_rejected(workflow, content):
    workflow.pending[1] = response(content=content)
    with pytest.raises(RuntimeError, match="有效正文"):
        workflow.module.main(workflow.csv)


def test_four_tool_schemas_do_not_change_task08_tool_definitions(workflow):
    from examples.tool_calling import TOOLS as observation_tools

    functions = {
        tool["function"]["name"]: tool["function"] for tool in workflow.registry.TOOLS
    }
    assert set(functions) == {
        "dataset_info",
        "describe_numeric",
        "group_aggregate",
        "correlation",
    }
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
    json.dumps(workflow.registry.TOOLS, allow_nan=False)


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
    assert second["tools"] == workflow.registry.TOOLS
    assert second["tool_choice"] == "auto"
    assert "response_format" not in second
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
        workflow.service.execute_tool(frame, "dataset_info", '{"columns":["sales"]}')
    workflow.execute.assert_not_called()
    workflow.describe.assert_not_called()


def test_model_cannot_request_unregistered_tool_with_valid_arguments(workflow):
    frame = real_load_csv(workflow.csv)
    with pytest.raises(ValueError, match="工具名称"):
        workflow.service.execute_tool(frame, "unknown_tool", "{}")
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
        workflow.service.execute_tool(frame, "describe_numeric", '{"columns":"sales"}')
    workflow.describe.assert_not_called()


def test_group_schema_has_exact_required_arguments_and_enum(workflow):
    tools = {
        tool["function"]["name"]: tool["function"] for tool in workflow.registry.TOOLS
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
    assert second["tools"] == workflow.registry.TOOLS
    assert second["tool_choice"] == "auto"
    assert "response_format" not in second
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


def test_correlation_schema_requires_two_columns_and_pearson(workflow):
    functions = {
        tool["function"]["name"]: tool["function"] for tool in workflow.registry.TOOLS
    }
    parameters = functions["correlation"]["parameters"]
    fields = {"x_column", "y_column", "method"}
    assert parameters["type"] == "object"
    assert set(parameters["required"]) == fields
    assert set(parameters["properties"]) == fields
    assert parameters["additionalProperties"] is False
    for field in fields:
        assert parameters["properties"][field]["type"] == "string"
    assert parameters["properties"]["method"]["enum"] == ["pearson"]


def run_correlation_case(workflow, *, x_column="x", y_column="y"):
    """Run real CSV/statistics with only the model replies replaced."""
    arguments = json.dumps(
        {"x_column": x_column, "y_column": y_column, "method": "pearson"}
    )
    workflow.pending[0] = response(
        calls=[
            tool_call(
                name="correlation",
                identifier="correlation_id",
                arguments=arguments,
            )
        ],
        finish_reason="tool_calls",
    )
    workflow.pending[1] = response(content="模拟相关性解释。")
    question = "请计算这两个数值字段的 Pearson 相关系数并说明有效样本数"
    before = real_load_csv(workflow.csv)
    workflow.module.main(workflow.csv, question)
    workflow.execute.assert_not_called()
    workflow.describe.assert_not_called()
    workflow.aggregate.assert_not_called()
    workflow.correlate.assert_called_once()
    frame = workflow.correlate.call_args.args[0]
    pd.testing.assert_frame_equal(frame, before)
    assert workflow.correlate.call_args.kwargs == {
        "x_column": x_column,
        "y_column": y_column,
        "method": "pearson",
    }
    assert workflow.events == ["request", "execute", "request"]
    first, second = workflow.requests
    assert first["messages"][1] == {"role": "user", "content": question}
    system = first["messages"][0]["content"]
    assert json.loads(system[system.index("{") :]) == {
        name: str(dtype) for name, dtype in before.dtypes.items()
    }
    assert second["messages"][:2] == first["messages"]
    assistant, tool = second["messages"][2:]
    assert assistant["tool_calls"] == [
        {
            "id": "correlation_id",
            "type": "function",
            "function": {"name": "correlation", "arguments": arguments},
        }
    ]
    assert tool["role"] == "tool" and tool["tool_call_id"] == "correlation_id"
    assert second["tools"] == workflow.registry.TOOLS
    assert second["tool_choice"] == "auto"
    assert "response_format" not in second
    result = json.loads(tool["content"])
    assert set(result) == {
        "x_column",
        "y_column",
        "method",
        "coefficient",
        "sample_size",
        "status",
    }
    assert result["x_column"] == x_column and result["y_column"] == y_column
    assert result["method"] == "pearson"
    json.dumps(result, allow_nan=False)
    return result


@pytest.mark.parametrize(
    "values,expected",
    [
        ("1,2\n2,4\n3,6\n", 1.0),
        ("1,6\n2,4\n3,2\n", -1.0),
        ("1,1\n2,0\n3,1\n", 0.0),
    ],
)
def test_known_correlations_are_returned_including_genuine_zero(
    workflow, values, expected
):
    workflow.csv.write_text("x,y\n" + values, encoding="utf-8")
    result = run_correlation_case(workflow)
    assert result["status"] == "ok"
    assert result["sample_size"] == 3
    assert result["coefficient"] == pytest.approx(expected)


def test_correlation_counts_complete_pairs_not_individual_values(workflow):
    workflow.csv.write_text(
        "x,y,region\n1,2,A\n2,,B\n3,6,C\n,8,D\n5,10,E\n,,F\n", encoding="utf-8"
    )
    df = real_load_csv(workflow.csv)
    assert int(df["x"].count()) == int(df["y"].count()) == 4
    result = run_correlation_case(workflow)
    assert result["sample_size"] == 3
    assert result["coefficient"] == pytest.approx(1.0)
    assert result["status"] == "ok"


def test_same_numeric_column_is_allowed_by_existing_tool(workflow):
    workflow.csv.write_text("x,y\n1,2\n2,4\n3,6\n", encoding="utf-8")
    result = run_correlation_case(workflow, y_column="x")
    assert result["status"] == "ok"
    assert result["coefficient"] == pytest.approx(1.0)


@pytest.mark.parametrize("values", ["1,2\n1,3\n1,4\n", "1,2\n2,2\n3,2\n"])
def test_constant_column_returns_zero_variance_with_null(workflow, values):
    workflow.csv.write_text("x,y\n" + values, encoding="utf-8")
    result = run_correlation_case(workflow)
    assert result["coefficient"] is None
    assert result["sample_size"] == 3
    assert result["status"] == "zero_variance"


@pytest.mark.parametrize("values,n", [("1,2\n2,\n", 1), ("1,\n,2\n", 0)])
def test_insufficient_complete_pairs_return_null_not_zero(workflow, values, n):
    workflow.csv.write_text("x,y\n" + values, encoding="utf-8")
    result = run_correlation_case(workflow)
    assert result["coefficient"] is None
    assert result["sample_size"] == n
    assert result["status"] == "insufficient_samples"


def test_sales_correlation_matches_independent_statistics_calculation(workflow):
    workflow.csv = Path(__file__).resolve().parents[1] / "data/samples/sales.csv"
    with workflow.csv.open(encoding="utf-8", newline="") as source:
        pairs = [
            (float(row["sales"]), float(row["advertising_spend"]))
            for row in csv.DictReader(source)
            if row["sales"] and row["advertising_spend"]
        ]
    assert len(pairs) == 36
    expected = statistics.correlation(*zip(*pairs, strict=True))
    result = run_correlation_case(
        workflow, x_column="sales", y_column="advertising_spend"
    )
    assert result["sample_size"] == 36
    assert result["status"] == "ok"
    assert result["coefficient"] == pytest.approx(expected)
    assert result["coefficient"] == pytest.approx(0.9409742630262548)


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"x_column": "sales", "y_column": "sales"},
        {"x_column": "sales", "method": "pearson"},
        {"y_column": "sales", "method": "pearson"},
        {"x_column": "sales", "y_column": "sales", "method": "pearson", "fillna": 0},
    ],
)
def test_correlation_requires_exact_fields(workflow, arguments):
    workflow.pending[0] = response(
        calls=[
            tool_call(
                name="correlation",
                arguments=json.dumps(arguments),
            )
        ],
        finish_reason="tool_calls",
    )
    with pytest.raises(ValueError, match="三个必填参数"):
        workflow.module.main(workflow.csv)
    assert_no_execution_or_second_request(workflow)


@pytest.mark.parametrize("field", ["x_column", "y_column", "method"])
@pytest.mark.parametrize("value", [None, 1, True, []])
def test_correlation_parameter_types_are_strict(workflow, field, value):
    arguments = {"x_column": "sales", "y_column": "sales", "method": "pearson"}
    arguments[field] = value
    workflow.pending[0] = response(
        calls=[
            tool_call(
                name="correlation",
                arguments=json.dumps(arguments),
            )
        ],
        finish_reason="tool_calls",
    )
    with pytest.raises(ValueError, match="必须是字符串"):
        workflow.module.main(workflow.csv)
    assert_no_execution_or_second_request(workflow)


@pytest.mark.parametrize("method", ["spearman", "kendall", "Pearson", ""])
def test_correlation_method_is_checked_locally_despite_enum(workflow, method):
    arguments = {"x_column": "sales", "y_column": "sales", "method": method}
    workflow.pending[0] = response(
        calls=[
            tool_call(
                name="correlation",
                arguments=json.dumps(arguments),
            )
        ],
        finish_reason="tool_calls",
    )
    with pytest.raises(ValueError, match="Unsupported correlation method"):
        workflow.module.main(workflow.csv)
    workflow.correlate.assert_called_once()
    assert workflow.create.call_count == 1


@pytest.mark.parametrize("field", ["x_column", "y_column"])
@pytest.mark.parametrize("column", ["absent", "region", " "])
def test_both_correlation_columns_are_validated(workflow, field, column):
    arguments = {"x_column": "sales", "y_column": "sales", "method": "pearson"}
    arguments[field] = column
    workflow.pending[0] = response(
        calls=[
            tool_call(
                name="correlation",
                arguments=json.dumps(arguments),
            )
        ],
        finish_reason="tool_calls",
    )
    with pytest.raises(ValueError):
        workflow.module.main(workflow.csv)
    workflow.correlate.assert_called_once()
    assert workflow.create.call_count == 1


@pytest.mark.parametrize("arguments", ["not JSON", "[]", "null"])
def test_correlation_arguments_must_be_json_object(workflow, arguments):
    workflow.pending[0] = response(
        calls=[
            tool_call(
                name="correlation",
                arguments=arguments,
            )
        ],
        finish_reason="tool_calls",
    )
    expected = json.JSONDecodeError if arguments == "not JSON" else ValueError
    with pytest.raises(expected):
        workflow.module.main(workflow.csv)
    assert_no_execution_or_second_request(workflow)


def test_correlation_failure_prevents_second_request(workflow, capsys):
    original = RuntimeError("correlation calculation failed")
    workflow.correlate.side_effect = original
    arguments = '{"x_column":"sales","y_column":"sales","method":"pearson"}'
    workflow.pending[0] = response(
        calls=[
            tool_call(
                name="correlation",
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
