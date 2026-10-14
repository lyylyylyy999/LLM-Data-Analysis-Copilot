"""独立实验：使用 JSON Schema 约束模型输出，再交给 Python 校验。"""

import json
import os

from dotenv import load_dotenv

from data_copilot.llm_client import SYSTEM_PROMPT
from data_copilot.response_parser import parse_analysis_answer

# 手工声明请求 Schema；不依赖 Pydantic，也不修改 ask_llm()。
ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "explanation": {"type": "string"},
        "missing_information": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["summary", "explanation", "missing_information"],
    "additionalProperties": False,
}

QUESTIONS = (
    "什么是过拟合？",
    "请计算广告投入和销售额的相关系数。",
    "广告投入和销售额显著相关，是否说明广告导致销售增长？",
)


def ask_structured(question: str) -> dict:
    """单次非流式请求，返回通过本地校验的结构化回答。"""
    if not question.strip():
        raise ValueError("question 不能为空或空白内容")

    load_dotenv()
    api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    base_url = (os.getenv("DASHSCOPE_BASE_URL") or "").strip()
    model = (os.getenv("DASHSCOPE_MODEL") or "").strip()

    required_config = {
        "DASHSCOPE_API_KEY": api_key,
        "DASHSCOPE_BASE_URL": base_url,
        "DASHSCOPE_MODEL": model,
    }
    missing = [name for name, value in required_config.items() if not value]
    if missing:
        raise RuntimeError(f"缺少必要的环境变量：{', '.join(missing)}")

    # 与 llm_client.py 使用相同的客户端连接配置。
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "analysis_answer",
                "strict": True,
                "schema": ANALYSIS_SCHEMA,
            },
        },
        stream=False,
    )

    if not completion.choices:
        raise RuntimeError("模型未返回候选回答")

    choice = completion.choices[0]
    if choice.finish_reason == "length":
        raise RuntimeError("模型回答被截断")
    if choice.finish_reason != "stop":
        raise RuntimeError(f"模型未正常结束：{choice.finish_reason}")

    if getattr(choice.message, "refusal", None):
        raise RuntimeError("模型拒绝回答")

    # 不剥离代码围栏、不修补 JSON，也不回退到提示词模式。
    answer = choice.message.content
    if not answer or not answer.strip():
        raise RuntimeError("模型未返回有效正文")
    return parse_analysis_answer(answer)


def main() -> None:
    for question in QUESTIONS:
        parsed = ask_structured(question)
        print(
            json.dumps(
                {"question": question, "result": parsed},
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )


if __name__ == "__main__":
    main()
