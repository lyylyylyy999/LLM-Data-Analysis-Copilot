"""Offline tests of the HTTP-only frontend and its explicit submit boundary."""

import copy
import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest


@pytest.fixture
def frontend(monkeypatch):
    payload = {
        "answer": "模拟分析结果",
        "model": "actual-model",
        "finish_reason": "stop",
        "tool_results": [
            {
                "tool_call_id": "call_1",
                "name": "dataset_info",
                "arguments": {},
                "result": {"row_count": 3, "missing": None},
            }
        ],
    }
    post = Mock(return_value=httpx.Response(200, json=copy.deepcopy(payload)))
    monkeypatch.setattr(httpx, "post", post)
    forbidden = Mock(side_effect=AssertionError("No outbound HTTP allowed"))
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    path = Path(__file__).resolve().parents[1] / "frontend/app.py"
    spec = importlib.util.spec_from_file_location("frontend_offline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    post.assert_not_called()
    return SimpleNamespace(module=module, post=post, payload=payload)


def submit(frontend, **kwargs):
    arguments = {
        "api_url": "http://example.invalid/",
        "question": "  分析问题  ",
        "filename": "tiny.csv",
        "file_content": b"x\n1\n",
    }
    arguments.update(kwargs)
    return frontend.module.submit_analysis(**arguments)


def test_submit_sends_exact_multipart_once_with_explicit_timeout(frontend):
    assert submit(frontend) == frontend.payload
    frontend.post.assert_called_once_with(
        "http://example.invalid/analyze-file",
        data={"question": "  分析问题  "},
        files={"file": ("tiny.csv", b"x\n1\n", "text/csv")},
        timeout=frontend.module.REQUEST_TIMEOUT,
        trust_env=True,
    )
    timeout = frontend.module.REQUEST_TIMEOUT
    assert timeout.connect == 10
    assert timeout.read == 300
    assert timeout.write is not None and timeout.pool is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"question": ""},
        {"question": " \n\t"},
        {"filename": ""},
        {"filename": "tiny.txt"},
        {"filename": "tiny.csv.exe"},
        {"file_content": b"x" * (5 * 1024 * 1024 + 1)},
    ],
)
def test_invalid_input_is_rejected_without_http(frontend, kwargs):
    with pytest.raises(frontend.module.FrontendError):
        submit(frontend, **kwargs)
    frontend.post.assert_not_called()


def test_exact_size_limit_and_uppercase_extension_are_allowed(frontend):
    assert frontend.module.MAX_UPLOAD_BYTES == 5 * 1024 * 1024
    content = b"x" * frontend.module.MAX_UPLOAD_BYTES
    assert (
        submit(frontend, filename="tiny.CSV", file_content=content) == frontend.payload
    )
    assert frontend.post.call_args.kwargs["files"]["file"][1] is content


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://[::1]:8000",
    ],
)
def test_local_backend_bypasses_proxy_environment(frontend, monkeypatch, url):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:7890")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:7890")
    monkeypatch.setenv("ALL_PROXY", "http://proxy.invalid:7890")
    monkeypatch.delenv("NO_PROXY", raising=False)
    assert submit(frontend, api_url=url) == frontend.payload
    assert frontend.post.call_args.kwargs["trust_env"] is False


def test_remote_backend_preserves_proxy_support(frontend):
    submit(frontend, api_url="https://remote.invalid")
    assert frontend.post.call_args.kwargs["trust_env"] is True


@pytest.mark.parametrize(
    "status,detail",
    [
        (502, "模型服务调用失败"),
        (504, "模型服务响应超时"),
    ],
)
def test_known_backend_gateway_errors_keep_model_explanation(frontend, status, detail):
    frontend.post.return_value = httpx.Response(status, json={"detail": detail})
    with pytest.raises(frontend.module.FrontendError) as caught:
        submit(frontend)
    assert str(caught.value) == f"{detail}，请稍后重试。"


@pytest.mark.parametrize("payload", [None, [], {"detail": "proxy connection failed"}])
def test_unknown_proxy_502_is_not_misattributed_to_model(frontend, payload):
    frontend.post.return_value = httpx.Response(502, json=payload)
    with pytest.raises(frontend.module.FrontendError) as caught:
        submit(frontend)
    assert "代理网关" in str(caught.value)
    assert "模型服务调用失败" not in str(caught.value)


@pytest.mark.parametrize("status", [400, 413, 422, 500, 502, 504, 401, 302])
def test_http_errors_are_safe_messages_and_never_analysis_results(frontend, status):
    frontend.post.return_value = httpx.Response(
        status, text="fake-secret-upstream-detail"
    )
    with pytest.raises(frontend.module.FrontendError) as caught:
        submit(frontend)
    expected = frontend.module.HTTP_ERROR_MESSAGES.get(
        status, f"后端请求失败（HTTP {status}）。"
    )
    assert str(caught.value) == expected
    assert "fake-secret" not in str(caught.value)
    frontend.post.assert_called_once()


@pytest.mark.parametrize(
    "exception,expected",
    [
        (httpx.ReadTimeout("private timeout detail"), "请求超时"),
        (httpx.ConnectTimeout("private connect detail"), "请求超时"),
        (httpx.ConnectError("private connection detail"), "无法连接后端"),
        (httpx.ReadError("private read detail"), "与后端通信失败"),
    ],
)
def test_transport_failures_are_distinct_and_preserve_cause(
    frontend, exception, expected
):
    frontend.post.side_effect = exception
    with pytest.raises(frontend.module.FrontendError, match=expected) as caught:
        submit(frontend)
    assert caught.value.__cause__ is exception
    assert "private" not in str(caught.value)
    frontend.post.assert_called_once()


@pytest.mark.parametrize("content", ["", "<html>Error</html>", "{broken"])
def test_non_json_success_is_rejected(frontend, content):
    frontend.post.return_value = httpx.Response(200, text=content)
    with pytest.raises(frontend.module.FrontendError, match="JSON"):
        submit(frontend)


@pytest.mark.parametrize(
    "field,value",
    [
        ("answer", None),
        ("answer", " \n"),
        ("model", 42),
        ("model", ""),
        ("finish_reason", "length"),
        ("tool_results", {}),
        ("tool_results", [None]),
        ("tool_results", [{}]),
    ],
)
def test_malformed_success_payload_is_rejected(frontend, field, value):
    payload = copy.deepcopy(frontend.payload)
    payload[field] = value
    frontend.post.return_value = httpx.Response(200, json=payload)
    with pytest.raises(frontend.module.FrontendError, match="响应结构异常"):
        submit(frontend)


@pytest.mark.parametrize("payload", [None, [], "text", {}, {"answer": "text"}])
def test_non_protocol_response_is_rejected(frontend, payload):
    frontend.post.return_value = httpx.Response(200, json=payload)
    with pytest.raises(frontend.module.FrontendError):
        submit(frontend)


@pytest.mark.parametrize(
    "field,value",
    [
        ("tool_call_id", ""),
        ("name", " \t"),
        ("arguments", []),
        ("result", None),
    ],
)
def test_invalid_nested_result_is_rejected(frontend, field, value):
    payload = copy.deepcopy(frontend.payload)
    payload["tool_results"][0][field] = value
    frontend.post.return_value = httpx.Response(200, json=payload)
    with pytest.raises(frontend.module.FrontendError, match="工具结果"):
        submit(frontend)


@pytest.fixture
def page(frontend, monkeypatch):
    uploaded = SimpleNamespace(
        name="tiny.csv", size=4, getvalue=Mock(return_value=b"x\n1\n")
    )
    st = SimpleNamespace(
        set_page_config=Mock(),
        title=Mock(),
        caption=Mock(),
        form=Mock(side_effect=lambda *a, **k: nullcontext()),
        file_uploader=Mock(return_value=uploaded),
        text_area=Mock(return_value="问题"),
        form_submit_button=Mock(return_value=True),
        error=Mock(),
        spinner=Mock(side_effect=lambda *a, **k: nullcontext()),
        subheader=Mock(),
        markdown=Mock(),
        expander=Mock(side_effect=lambda *a, **k: nullcontext()),
        write=Mock(),
        json=Mock(),
    )
    monkeypatch.setitem(sys.modules, "streamlit", st)
    monkeypatch.delenv("DATA_COPILOT_API_URL", raising=False)
    return SimpleNamespace(st=st, upload=uploaded)


def test_reruns_without_submit_do_not_read_file_or_request(frontend, page):
    page.st.form_submit_button.return_value = False
    frontend.module.main()
    frontend.module.main()
    page.upload.getvalue.assert_not_called()
    frontend.post.assert_not_called()
    page.st.error.assert_not_called()


@pytest.mark.parametrize(
    "problem", ["no_file", "blank_question", "oversize", "blank_url"]
)
def test_page_preconditions_prevent_file_read_and_http(
    frontend, page, monkeypatch, problem
):
    if problem == "no_file":
        page.st.file_uploader.return_value = None
    elif problem == "blank_question":
        page.st.text_area.return_value = " \n"
    elif problem == "oversize":
        page.upload.size = frontend.module.MAX_UPLOAD_BYTES + 1
    else:
        monkeypatch.setenv("DATA_COPILOT_API_URL", " ")
    frontend.module.main()
    page.st.error.assert_called_once()
    page.upload.getvalue.assert_not_called()
    frontend.post.assert_not_called()


def test_one_submit_requests_once_and_renders_without_unsafe_html(frontend, page):
    frontend.module.main()
    frontend.post.assert_called_once()
    assert frontend.post.call_args.args[0] == "http://127.0.0.1:8000/analyze-file"
    page.upload.getvalue.assert_called_once()
    page.st.markdown.assert_called_once_with(frontend.payload["answer"])
    page.st.json.assert_called_once_with(frontend.payload["tool_results"])
    page.st.write.assert_any_call("模型：", frontend.payload["model"])
    page.st.write.assert_any_call("结束原因：", "stop")
    page.st.error.assert_not_called()
    page.st.form_submit_button.return_value = False
    frontend.module.main()
    frontend.post.assert_called_once()


def test_page_uses_configured_backend(frontend, page, monkeypatch):
    monkeypatch.setenv("DATA_COPILOT_API_URL", " http://configured.invalid/api/ ")
    frontend.module.main()
    assert (
        frontend.post.call_args.args[0] == "http://configured.invalid/api/analyze-file"
    )


@pytest.mark.parametrize(
    "failure", ["status", "nonjson", "malformed", "connection", "timeout"]
)
def test_page_displays_errors_without_rendering_analysis(frontend, page, failure):
    if failure == "status":
        frontend.post.return_value = httpx.Response(502)
    elif failure == "nonjson":
        frontend.post.return_value = httpx.Response(200, text="not JSON")
    elif failure == "malformed":
        frontend.post.return_value = httpx.Response(200, json={})
    else:
        frontend.post.side_effect = (
            httpx.ConnectError("offline")
            if failure == "connection"
            else httpx.ReadTimeout("offline")
        )
    frontend.module.main()
    page.st.error.assert_called_once()
    page.st.markdown.assert_not_called()
    page.st.json.assert_not_called()
