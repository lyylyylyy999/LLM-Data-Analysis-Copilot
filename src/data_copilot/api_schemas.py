"""HTTP 分析接口的请求与响应约束。"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator


class AnalyzeRequest(BaseModel):
    """只允许提交一个非空白的分析问题。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    question: str

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question 不能为空或仅包含空白字符")
        return value


class ToolExecutionResult(BaseModel):
    """单次工具调用的 ID、参数与实际结果。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]

    @field_validator("tool_call_id", "name")
    @classmethod
    def nonblank_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("工具调用 ID 和工具名称不能为空或仅包含空白字符")
        return value


class AnalyzeResponse(BaseModel):
    """成功分析返回的固定字段；统计结果保留为通用字典。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    answer: str
    model: str
    finish_reason: Literal["stop"]
    tool_results: list[ToolExecutionResult]

    @field_validator("answer", "model")
    @classmethod
    def nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer 和 model 不能为空或仅包含空白字符")
        return value
