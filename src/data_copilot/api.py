"""最小 HTTP 分析接口：固定示例 CSV，复用既有同步分析服务。"""

from pathlib import Path
from typing import Any

from fastapi import FastAPI

from data_copilot.analysis_service import analyze_question
from data_copilot.api_schemas import AnalyzeRequest, AnalyzeResponse
from data_copilot.data_loader import load_csv

# 从源文件位置定位项目根目录，不依赖 uvicorn 的启动目录。
SAMPLE_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "samples" / "sales.csv"

app = FastAPI(title="Data Analysis Copilot")


@app.get("/health")
def health() -> dict[str, str]:
    """独立健康检查，不读取数据、不访问模型。"""
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    """每个有效请求重新读取固定 CSV，响应由 Pydantic 校验。"""
    dataframe = load_csv(SAMPLE_CSV_PATH)
    return analyze_question(dataframe, request.question)
