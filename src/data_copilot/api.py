"""HTTP 分析接口：固定示例数据或临时上传 CSV，复用同步分析服务。"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.responses import Response

from data_copilot.analysis_service import analyze_question
from data_copilot.api_schemas import AnalyzeRequest, AnalyzeResponse
from data_copilot.data_loader import load_csv

# 不依赖启动 uvicorn 时的当前目录。
SAMPLE_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "samples" / "sales.csv"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

app = FastAPI(title="Data Analysis Copilot")


@app.exception_handler(RequestValidationError)
async def request_validation_error(
    request: Request, exc: RequestValidationError
) -> Response:
    """将 multipart 中缺失 filename 的文件字段判定为文件名错误。"""
    errors = exc.errors()
    if request.url.path == "/analyze-file" and not any(
        error.get("loc") == ("body", "question") for error in errors
    ):
        for error in errors:
            if (
                error.get("loc") == ("body", "file")
                and error.get("type") == "value_error"
                and isinstance(error.get("input"), str)
            ):
                return JSONResponse(
                    status_code=400, content={"detail": "上传文件缺少文件名"}
                )
    return await request_validation_exception_handler(request, exc)


@app.get("/health")
def health() -> dict[str, str]:
    """健康检查不读取数据，也不访问模型。"""
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    """使用固定示例 CSV，响应交由 Pydantic 校验。"""
    dataframe = load_csv(SAMPLE_CSV_PATH)
    return analyze_question(dataframe, request.question)


@app.post("/analyze-file", response_model=AnalyzeResponse)
def analyze_file(
    file: Annotated[UploadFile, File(description="UTF-8 CSV，最大 5 MiB")],
    question: Annotated[str, Form(description="分析问题")],
) -> dict[str, Any]:
    """限制上传大小，临时读取用户 CSV，再交给原有分析服务。"""
    # 先验证问题；无效问题不读取上传文件，也不调用分析服务。
    if not question.strip():
        raise HTTPException(status_code=422, detail="question 不能为空或仅包含空白字符")

    # 上传文件名仅用于扩展名校验，绝不作为磁盘保存路径。
    if not file.filename or Path(file.filename).suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="上传文件必须具有 .csv 扩展名")

    # 只读取上限加一个字节，既能判断超限，也不无条件加载整个上传内容。
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="CSV 文件不得超过 5 MiB")

    # 使用服务端创建的固定临时文件名，退出 with 时自动清理（包括异常情况）。
    with TemporaryDirectory(prefix="data_copilot_upload_") as directory:
        path = Path(directory) / "uploaded.csv"
        path.write_bytes(content)
        try:
            # load_csv 默认使用 UTF-8（兼容 BOM），保留原有 CSV 校验规则。
            dataframe = load_csv(path)
        except (UnicodeError, ValueError) as exc:
            # 仅转换 CSV 读取过程中的已知输入错误，不捕获后续分析错误。
            raise HTTPException(status_code=400, detail=f"无效 CSV：{exc}") from exc
        return analyze_question(dataframe, question)
