"""最小 HTTP 分析接口：固定示例 CSV，复用既有同步分析服务。"""

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, field_validator

from data_copilot.analysis_service import analyze_question
from data_copilot.data_loader import load_csv

# 从源文件位置定位项目根目录，不依赖 uvicorn 的启动目录。
SAMPLE_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "samples" / "sales.csv"


class AnalyzeRequest(BaseModel):
    """分析请求只允许提交一个非空白问题。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    question: str

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question 不能为空或仅包含空白字符")
        return value


app = FastAPI(title="Data Analysis Copilot")


@app.get("/health")
def health() -> dict[str, str]:
    """独立健康检查，不读取数据、不访问模型。"""
    return {"status": "ok"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    """每个有效请求重新读取固定 CSV，直接返回服务层的 JSON 结果。"""
    dataframe = load_csv(SAMPLE_CSV_PATH)
    return analyze_question(dataframe, request.question)
