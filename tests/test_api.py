"""Offline HTTP boundary tests; no real model requests or secrets are used."""

import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from data_copilot.data_loader import load_csv as real_load_csv


@pytest.fixture
def api(monkeypatch):
    forbidden = Mock(
        side_effect=AssertionError("API tests must not access LLM or network")
    )
    # TestClient runs in process; Windows asyncio needs a local socketpair.
    # Block outbound HTTP transports without breaking that internal mechanism.
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    monkeypatch.setattr("dotenv.load_dotenv", forbidden)
    monkeypatch.setattr("openai.OpenAI", forbidden)
    path = Path(__file__).resolve().parents[1] / "src/data_copilot/api.py"
    spec = importlib.util.spec_from_file_location("api_offline_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = pd.DataFrame({"sales": [10, None, 30]})
    payload = {
        "answer": "模拟分析回答",
        "model": "actual-response-model",
        "finish_reason": "stop",
        "tool_results": [
            {
                "tool_call_id": "call_1",
                "name": "describe_numeric",
                "arguments": {"columns": ["sales"]},
                "result": {"mean": 20.0, "unavailable": None},
            }
        ],
    }
    load = Mock(return_value=frame)
    service = Mock(return_value=copy.deepcopy(payload))
    monkeypatch.setattr(module, "load_csv", load)
    monkeypatch.setattr(module, "analyze_question", service)
    with TestClient(module.app) as client:
        yield SimpleNamespace(
            module=module,
            client=client,
            load=load,
            service=service,
            frame=frame,
            payload=payload,
            forbidden=forbidden,
        )
    forbidden.assert_not_called()


def test_health_does_not_load_data_or_call_service(api):
    response = api.client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    api.load.assert_not_called()
    api.service.assert_not_called()


def test_analyze_delegates_dataframe_and_preserves_json(api):
    question = "  每个地区的平均销售额？  "
    response = api.client.post("/analyze", json={"question": question})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == api.payload
    api.load.assert_called_once_with(api.module.SAMPLE_CSV_PATH)
    api.service.assert_called_once()
    assert api.service.call_args.args[0] is api.frame
    assert api.service.call_args.args[1] == question
    json.dumps(response.json(), ensure_ascii=False, allow_nan=False)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": ""},
        {"question": " \t\n"},
        {"question": None},
        {"question": 42},
        {"question": True},
        {"question": []},
        {"question": {}},
        {"question": "问题", "csv_path": "other.csv"},
        {"question": "问题", "model": "other-model"},
        {"question": "问题", "api_key": "fake-key"},
        {"question": "问题", "unexpected": 1},
        [],
        "问题",
        42,
    ],
)
def test_invalid_request_is_422_before_data_or_service(api, body):
    response = api.client.post("/analyze", json=body)
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    api.load.assert_not_called()
    api.service.assert_not_called()


@pytest.mark.parametrize("content", [None, "{", '{"question":'])
def test_absent_or_invalid_json_is_422(api, content):
    response = api.client.post(
        "/analyze", content=content, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    api.load.assert_not_called()
    api.service.assert_not_called()


def test_each_request_loads_fresh_data(api):
    second = pd.DataFrame({"sales": [99]})
    api.load.side_effect = [api.frame, second]
    for question in ("问题一", "问题二"):
        assert (
            api.client.post("/analyze", json={"question": question}).status_code == 200
        )
    assert api.load.call_count == api.service.call_count == 2
    assert api.service.call_args_list[0].args[0] is api.frame
    assert api.service.call_args_list[1].args[0] is second


def test_fixed_sample_path_is_independent_of_cwd(api, monkeypatch, tmp_path):
    expected = Path(__file__).resolve().parents[1] / "data/samples/sales.csv"
    monkeypatch.chdir(tmp_path)
    assert api.module.SAMPLE_CSV_PATH == expected
    api.load.side_effect = real_load_csv
    response = api.client.post("/analyze", json={"question": "多少行？"})
    assert response.status_code == 200
    frame = api.service.call_args.args[0]
    assert frame.shape == (40, 6)
    assert frame["sales"].isna().sum() == 2


@pytest.mark.parametrize("stage", ["load", "service"])
def test_errors_propagate_without_success_payload(api, stage):
    original = RuntimeError("offline failure")
    getattr(api, stage).side_effect = original
    with pytest.raises(RuntimeError) as caught:
        api.client.post("/analyze", json={"question": "问题"})
    assert caught.value is original
    if stage == "load":
        api.service.assert_not_called()
    with TestClient(api.module.app, raise_server_exceptions=False) as client:
        response = client.post("/analyze", json={"question": "问题"})
    assert response.status_code == 500


def test_api_import_has_no_data_config_client_or_output(monkeypatch, capsys):
    forbidden = Mock(side_effect=AssertionError("Import must not perform work"))
    from data_copilot import analysis_service, data_loader

    monkeypatch.setattr(data_loader, "load_csv", forbidden)
    monkeypatch.setattr(analysis_service, "analyze_question", forbidden)
    monkeypatch.setattr("dotenv.load_dotenv", forbidden)
    monkeypatch.setattr("openai.OpenAI", forbidden)
    original_getenv = os.getenv

    def guarded_getenv(name, default=None):
        if name.startswith("DASHSCOPE_"):
            return forbidden(name, default)
        return original_getenv(name, default)

    monkeypatch.setattr("os.getenv", guarded_getenv)
    path = Path(__file__).resolve().parents[1] / "src/data_copilot/api.py"
    spec = importlib.util.spec_from_file_location("api_import_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    forbidden.assert_not_called()
    assert capsys.readouterr() == ("", "")


def test_docs_are_available_without_analysis(api):
    assert api.client.get("/docs").status_code == 200
    schema = api.client.get("/openapi.json").json()
    assert "/health" in schema["paths"] and "/analyze" in schema["paths"]
    api.load.assert_not_called()
    api.service.assert_not_called()
