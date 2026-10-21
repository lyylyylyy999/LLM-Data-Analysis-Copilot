"""Streamlit 前端：仅通过 HTTP 上传 CSV 并展示分析结果。"""

import os
from pathlib import Path
from typing import Any

import httpx

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
DEFAULT_API_URL = "http://127.0.0.1:8000"
# 后端可能执行多批工具、调用多次模型；读取超时需覆盖整次分析。
REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)

HTTP_ERROR_MESSAGES = {
    400: "CSV 文件无效，请检查文件名、UTF-8 编码及文件内容。",
    413: "上传文件超过后端的 5 MiB 限制。",
    422: "提交的请求无效，请检查问题和文件字段。",
    500: "分析服务内部错误，请稍后重试。",
    502: "后端或代理网关返回 HTTP 502，请检查后端和代理设置。",
    504: "后端或代理网关返回 HTTP 504，请检查后端和代理设置。",
}


class FrontendError(Exception):
    """前端可以安全展示给用户的请求或响应错误。"""


def validate_analysis_response(payload: object) -> dict[str, Any]:
    """验证后端返回的最小协议，避免意外响应导致页面异常。"""
    required = {"answer", "model", "finish_reason", "tool_results"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise FrontendError("后端响应结构异常：缺少或存在额外的结果字段。")

    answer = payload["answer"]
    model = payload["model"]
    if not isinstance(answer, str) or not answer.strip():
        raise FrontendError("后端响应结构异常：answer 不是非空字符串。")
    if not isinstance(model, str) or not model.strip():
        raise FrontendError("后端响应结构异常：model 不是非空字符串。")
    if payload["finish_reason"] != "stop":
        raise FrontendError("后端响应结构异常：模型未正常完成回答。")

    tool_results = payload["tool_results"]
    if not isinstance(tool_results, list):
        raise FrontendError("后端响应结构异常：tool_results 必须是列表。")

    fields = {"tool_call_id", "name", "arguments", "result"}
    for index, item in enumerate(tool_results, start=1):
        if not isinstance(item, dict) or set(item) != fields:
            raise FrontendError(f"后端响应结构异常：工具结果 {index} 字段不完整。")
        if (
            not all(
                isinstance(item[key], str) and item[key].strip()
                for key in ("tool_call_id", "name")
            )
            or not isinstance(item["arguments"], dict)
            or not isinstance(item["result"], dict)
        ):
            raise FrontendError(f"后端响应结构异常：工具结果 {index} 类型错误。")
    return payload


def submit_analysis(
    *, api_url: str, question: str, filename: str, file_content: bytes
) -> dict[str, Any]:
    """校验上传并向后端发一次 multipart 请求；不包含 Streamlit 逻辑。"""
    if not question.strip():
        raise FrontendError("请输入非空白的分析问题。")
    if not filename or Path(filename).suffix.lower() != ".csv":
        raise FrontendError("请选择 .csv 格式的文件。")
    if len(file_content) > MAX_UPLOAD_BYTES:
        raise FrontendError("CSV 文件不得超过 5 MiB。")

    # 本机前后端通信直接连接，避免环境代理代替停服的后端返回网关错误。
    local_backend = httpx.URL(api_url).host in {"127.0.0.1", "localhost", "::1"}
    try:
        response = httpx.post(
            f"{api_url.rstrip('/')}/analyze-file",
            data={"question": question},
            files={"file": (filename, file_content, "text/csv")},
            timeout=REQUEST_TIMEOUT,
            trust_env=not local_backend,
        )
    except httpx.TimeoutException as exc:
        raise FrontendError("请求超时；后端可能正在进行多轮模型分析。") from exc
    except httpx.ConnectError as exc:
        raise FrontendError("无法连接后端，请确认 FastAPI 已启动且地址正确。") from exc
    except httpx.RequestError as exc:
        raise FrontendError("与后端通信失败，请检查网络和服务状态。") from exc

    if not response.is_success:
        message = HTTP_ERROR_MESSAGES.get(
            response.status_code,
            f"后端请求失败（HTTP {response.status_code}）。",
        )
        if response.status_code in {502, 504}:
            # 状态码本身不能证明错误来自模型服务；只识别后端约定的响应。
            try:
                error_payload = response.json()
            except ValueError:
                error_payload = None
            backend_detail = {
                502: "模型服务调用失败",
                504: "模型服务响应超时",
            }[response.status_code]
            if error_payload == {"detail": backend_detail}:
                message = f"{backend_detail}，请稍后重试。"
        raise FrontendError(message)

    try:
        payload = response.json()
    except ValueError as exc:
        raise FrontendError("后端未返回有效的 JSON 数据。") from exc
    return validate_analysis_response(payload)


def main() -> None:
    """只有用户点击提交按钮时才读取文件、发送 HTTP 请求。"""
    import streamlit as st

    st.set_page_config(page_title="数据分析助手", page_icon="📊")
    st.title("数据分析助手")
    st.caption("上传 UTF-8 CSV，输入问题，由后端执行真实统计并返回分析结果。")

    with st.form("analyze_csv_form"):
        uploaded_file = st.file_uploader("选择 CSV 文件（最大 5 MiB）", type=["csv"])
        question = st.text_area(
            "分析问题",
            placeholder="例如：每个地区的平均销售额是多少？",
        )
        submitted = st.form_submit_button("开始分析")

    if not submitted:
        return
    if uploaded_file is None:
        st.error("请先上传 CSV 文件。")
        return
    if not question.strip():
        st.error("请输入非空白的分析问题。")
        return
    if uploaded_file.size > MAX_UPLOAD_BYTES:
        st.error("CSV 文件不得超过 5 MiB。")
        return

    api_url = os.getenv("DATA_COPILOT_API_URL", DEFAULT_API_URL).strip()
    if not api_url:
        st.error("DATA_COPILOT_API_URL 不能为空。")
        return

    with st.spinner("正在上传并分析数据，可能需要等待多轮模型调用……"):
        try:
            result = submit_analysis(
                api_url=api_url,
                question=question,
                filename=uploaded_file.name,
                file_content=uploaded_file.getvalue(),
            )
        except FrontendError as exc:
            st.error(str(exc))
            return

    st.subheader("分析结果")
    # 不开启 unsafe_allow_html，避免将模型正文作为不安全 HTML 渲染。
    st.markdown(result["answer"])
    with st.expander("模型与工具执行详情"):
        st.write("模型：", result["model"])
        st.write("结束原因：", result["finish_reason"])
        st.json(result["tool_results"])


if __name__ == "__main__":
    main()
