"""Offline HTTP boundary tests; no real model requests or secrets are used."""

import copy
import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pandas as pd
import pytest
from fastapi import HTTPException, UploadFile
from fastapi.exceptions import ResponseValidationError
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


def test_openapi_exposes_request_response_and_nested_tool_models(api):
    schema = api.client.get("/openapi.json").json()
    operation = schema["paths"]["/analyze"]["post"]
    assert operation["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AnalyzeRequest"
    }
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AnalyzeResponse"
    }
    models = schema["components"]["schemas"]
    fields = {
        "AnalyzeRequest": {"question"},
        "AnalyzeResponse": {"answer", "model", "finish_reason", "tool_results"},
        "ToolExecutionResult": {"tool_call_id", "name", "arguments", "result"},
    }
    for name, expected in fields.items():
        assert set(models[name]["properties"]) == expected
        assert set(models[name]["required"]) == expected
        assert models[name]["additionalProperties"] is False
    assert models["AnalyzeResponse"]["properties"]["tool_results"] == {
        "type": "array",
        "items": {"$ref": "#/components/schemas/ToolExecutionResult"},
        "title": "Tool Results",
    }
    api.load.assert_not_called()
    api.service.assert_not_called()


def assert_invalid_service_response(api, location):
    # TestClient normally re-raises server errors, which distinguishes these
    # from RequestValidationError (422). Also verify the actual HTTP boundary.
    with pytest.raises(ResponseValidationError) as caught:
        api.client.post("/analyze", json={"question": "一个有效问题"})
    assert any(tuple(error["loc"]) == location for error in caught.value.errors())
    api.service.assert_called_once()
    with TestClient(api.module.app, raise_server_exceptions=False) as client:
        response = client.post("/analyze", json={"question": "一个有效问题"})
    assert response.status_code == 500
    assert response.text == "Internal Server Error"


@pytest.mark.parametrize("field", ["answer", "model", "finish_reason", "tool_results"])
def test_missing_top_level_response_field_is_server_error(api, field):
    del api.service.return_value[field]
    assert_invalid_service_response(api, ("response", field))


@pytest.mark.parametrize("field", ["tool_call_id", "name", "arguments", "result"])
def test_missing_nested_response_field_is_server_error(api, field):
    del api.service.return_value["tool_results"][0][field]
    assert_invalid_service_response(api, ("response", "tool_results", 0, field))


@pytest.mark.parametrize("field", ["answer", "model"])
@pytest.mark.parametrize("value", [None, 42, True, [], {}, "", " \t\n"])
def test_invalid_response_text_is_server_error(api, field, value):
    api.service.return_value[field] = value
    assert_invalid_service_response(api, ("response", field))


@pytest.mark.parametrize("field", ["tool_call_id", "name"])
@pytest.mark.parametrize("value", [None, 42, True, [], {}, "", " \t\n"])
def test_invalid_tool_identity_is_server_error(api, field, value):
    api.service.return_value["tool_results"][0][field] = value
    assert_invalid_service_response(api, ("response", "tool_results", 0, field))


@pytest.mark.parametrize("value", [None, 42, "length", "tool_calls", "STOP", ""])
def test_finish_reason_must_be_stop(api, value):
    api.service.return_value["finish_reason"] = value
    assert_invalid_service_response(api, ("response", "finish_reason"))


@pytest.mark.parametrize("value", [None, {}, "tool", 42])
def test_tool_results_must_be_list(api, value):
    api.service.return_value["tool_results"] = value
    assert_invalid_service_response(api, ("response", "tool_results"))


@pytest.mark.parametrize("value", [None, "tool", 42, []])
def test_tool_results_items_must_be_objects(api, value):
    api.service.return_value["tool_results"] = [value]
    assert_invalid_service_response(api, ("response", "tool_results", 0))


@pytest.mark.parametrize("field", ["arguments", "result"])
@pytest.mark.parametrize("value", [None, "{}", 42, [], [["x", 1]]])
def test_tool_parameters_and_results_are_strict_dictionaries(api, field, value):
    api.service.return_value["tool_results"][0][field] = value
    assert_invalid_service_response(api, ("response", "tool_results", 0, field))


@pytest.mark.parametrize("nested", [False, True])
def test_response_extra_fields_are_rejected_instead_of_silently_dropped(api, nested):
    if nested:
        api.service.return_value["tool_results"][0]["unexpected"] = "extra"
        location = ("response", "tool_results", 0, "unexpected")
    else:
        api.service.return_value["unexpected"] = "extra"
        location = ("response", "unexpected")
    assert_invalid_service_response(api, location)


def test_response_allows_no_tool_calls_and_preserves_text(api):
    payload = {
        "answer": " 相关不代表因果。 ",
        "model": " actual-model ",
        "finish_reason": "stop",
        "tool_results": [],
    }
    api.service.return_value = payload
    response = api.client.post("/analyze", json={"question": "相关代表因果吗？"})
    assert response.status_code == 200
    assert response.json() == payload


def test_response_preserves_multiple_generic_statistic_results(api):
    payload = copy.deepcopy(api.payload)
    payload["tool_results"].append(
        {
            "tool_call_id": "call_2",
            "name": "correlation",
            "arguments": {"x_column": "x", "y_column": "y", "method": "pearson"},
            "result": {
                "coefficient": None,
                "sample_size": 1,
                "status": "insufficient_samples",
            },
        }
    )
    api.service.return_value = payload
    response = api.client.post("/analyze", json={"question": "问题"})
    assert response.status_code == 200
    assert response.json() == payload
    json.dumps(response.json(), allow_nan=False)


def test_schema_validation_does_not_claim_semantic_answer_quality(api):
    api.service.return_value["answer"] = "(!256)"
    response = api.client.post("/analyze", json={"question": "问题"})
    assert response.status_code == 200
    assert response.json()["answer"] == "(!256)"


def track_uploaded_csv(api):
    paths = []

    def load(path):
        paths.append(path)
        assert path.is_file()
        assert path.name == "uploaded.csv"
        return real_load_csv(path)

    api.load.side_effect = load
    return paths


@pytest.mark.parametrize("filename", ["tiny.csv", "tiny.CSV", "../../tiny.csv"])
def test_upload_uses_real_user_data_and_cleans_temp_file(api, filename):
    paths = track_uploaded_csv(api)
    content = b"x,y\n1,2\n1,3\n1,4\n"
    response = api.client.post(
        "/analyze-file",
        data={"question": "计算相关性"},
        files={"file": (filename, content, "text/csv")},
    )
    assert response.status_code == 200
    assert response.json() == api.payload
    frame, question = api.service.call_args.args
    assert frame.to_dict(orient="list") == {"x": [1, 1, 1], "y": [2, 3, 4]}
    assert question == "计算相关性"
    assert len(paths) == 1
    assert paths[0] != api.module.SAMPLE_CSV_PATH
    assert not paths[0].exists() and not paths[0].parent.exists()


@pytest.mark.parametrize("filename", ["tiny.txt", "tiny", "tiny.csv.exe", "", None])
def test_upload_invalid_filename_is_400_without_csv_or_service(api, filename):
    response = api.client.post(
        "/analyze-file",
        data={"question": "问题"},
        files={"file": (filename, b"x\n1\n", "text/csv")},
    )
    assert response.status_code == 400
    api.load.assert_not_called()
    api.service.assert_not_called()


@pytest.mark.parametrize("question", ["", " \n\t"])
def test_upload_blank_question_is_422_without_processing_csv(api, question):
    response = api.client.post(
        "/analyze-file",
        data={"question": question},
        files={"file": ("tiny.csv", b"x\n1\n", "text/csv")},
    )
    assert response.status_code == 422
    api.load.assert_not_called()
    api.service.assert_not_called()


@pytest.mark.parametrize("missing", ["file", "question"])
def test_upload_missing_required_field_is_422(api, missing):
    kwargs = {
        "data": {"question": "问题"},
        "files": {"file": ("tiny.csv", b"x\n1\n", "text/csv")},
    }
    kwargs.pop("files" if missing == "file" else "data")
    response = api.client.post("/analyze-file", **kwargs)
    assert response.status_code == 422
    api.load.assert_not_called()
    api.service.assert_not_called()


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"\n\n",
        b"x,x\n1,2\n",
        b"x,\n1,2\n",
        b"x,y\n1,2,3\n",
        b'x,y\n"unterminated,2\n',
        b"x\n\xff\n",
    ],
)
def test_invalid_csv_is_400_and_temp_file_is_removed(api, content):
    paths = track_uploaded_csv(api)
    response = api.client.post(
        "/analyze-file",
        data={"question": "问题"},
        files={"file": ("tiny.csv", content, "text/csv")},
    )
    assert response.status_code == 400
    assert "无效 CSV" in response.json()["detail"]
    api.service.assert_not_called()
    assert len(paths) == 1
    assert not paths[0].parent.exists()


@pytest.mark.parametrize("extra_byte", [False, True])
def test_upload_exact_size_boundary(api, extra_byte):
    assert api.module.MAX_UPLOAD_BYTES == 5 * 1024 * 1024
    content = b"x\n" + b"a" * (api.module.MAX_UPLOAD_BYTES - 2 + int(extra_byte))
    response = api.client.post(
        "/analyze-file",
        data={"question": "问题"},
        files={"file": ("tiny.csv", content, "text/csv")},
    )
    assert response.status_code == (413 if extra_byte else 200)
    if extra_byte:
        api.load.assert_not_called()
        api.service.assert_not_called()
    else:
        api.load.assert_called_once()
        api.service.assert_called_once()
        assert not api.load.call_args.args[0].parent.exists()


def test_upload_reads_at_most_limit_plus_one_byte(api):
    stream = Mock(wraps=io.BytesIO(b"x\n1\n"))
    upload = UploadFile(file=stream, filename="tiny.csv")
    assert api.module.analyze_file(upload, "问题") == api.payload
    stream.read.assert_called_once_with(api.module.MAX_UPLOAD_BYTES + 1)


def test_blank_question_prevents_even_file_content_read(api):
    stream = Mock(side_effect=AssertionError("Must not read invalid request"))
    upload = UploadFile(file=stream, filename="tiny.csv")
    with pytest.raises(HTTPException) as caught:
        api.module.analyze_file(upload, " \n")
    assert caught.value.status_code == 422
    stream.read.assert_not_called()
    api.load.assert_not_called()
    api.service.assert_not_called()


@pytest.mark.parametrize(
    "exception", [ValueError("analysis failed"), RuntimeError("model failed")]
)
def test_upload_analysis_error_is_not_csv_error_and_temp_is_cleaned(api, exception):
    paths = track_uploaded_csv(api)
    api.service.side_effect = exception
    with pytest.raises(type(exception)) as caught:
        api.client.post(
            "/analyze-file",
            data={"question": "问题"},
            files={"file": ("tiny.csv", b"x\n1\n", "text/csv")},
        )
    assert caught.value is exception
    assert len(paths) == 1 and not paths[0].parent.exists()
    with TestClient(api.module.app, raise_server_exceptions=False) as client:
        response = client.post(
            "/analyze-file",
            data={"question": "问题"},
            files={"file": ("tiny.csv", b"x\n1\n", "text/csv")},
        )
    assert response.status_code == 500
    assert all(not path.parent.exists() for path in paths)


def test_upload_os_error_remains_server_error_and_cleans_temp(api):
    paths = []
    original = PermissionError("cannot read temporary file")

    def fail(path):
        paths.append(path)
        raise original

    api.load.side_effect = fail
    with pytest.raises(PermissionError) as caught:
        api.client.post(
            "/analyze-file",
            data={"question": "问题"},
            files={"file": ("tiny.csv", b"x\n1\n", "text/csv")},
        )
    assert caught.value is original
    api.service.assert_not_called()
    assert len(paths) == 1 and not paths[0].parent.exists()
