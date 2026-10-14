"""数据分析助手的结构化回答模型（Pydantic v2）。"""

from pydantic import BaseModel, ConfigDict, field_validator


class AnalysisAnswer(BaseModel):
    """校验模型回答的三个必需字段及非空摘要规则。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    summary: str
    explanation: str
    missing_information: list[str]

    @field_validator("summary")
    @classmethod
    def check_summary_not_blank(cls, value: str) -> str:
        """只检查空白，不改写用户提供的原始文本。"""
        if not value.strip():
            raise ValueError("summary 不能为空或仅包含空白字符")
        return value
