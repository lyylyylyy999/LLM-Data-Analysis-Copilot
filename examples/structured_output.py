"""独立实验：由 OpenAI SDK 从 Pydantic 模型生成 Schema 并解析结果。"""

import json
import os

from dotenv import load_dotenv

from data_copilot.llm_client import SYSTEM_PROMPT
from data_copilot.schemas import AnalysisAnswer

QUESTIONS = (
    "什么是过拟合？",
    "请计算广告投入和销售额的相关系数。",
    "广告投入和销售额显著相关，是否说明广告导致销售增长？",
)


def ask_structured(question: str) -> dict:
    """使用接口级结构约束，返回通过 Pydantic 校验的回答字典。"""
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

    # 与 ask_llm() 使用相同的客户端配置；仅在实际调用时加载 SDK。
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
    completion = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        response_format=AnalysisAnswer,
    )

    # parse() 可能在返回前因截断、内容过滤或 Pydantic 校验失败而抛异常。
    # 不捕获这些异常，也不降级到 create() 或提示词模式。
    if not completion.choices:
        raise RuntimeError("模型未返回候选回答")

    choice = completion.choices[0]
    if choice.finish_reason == "length":
        raise RuntimeError("模型回答被截断")
    if choice.finish_reason != "stop":
        raise RuntimeError(f"模型未正常结束：{choice.finish_reason}")
    if getattr(choice.message, "refusal", None):
        raise RuntimeError("模型拒绝回答")

    parsed = choice.message.parsed
    if not isinstance(parsed, AnalysisAnswer):
        raise RuntimeError("模型未返回有效解析结果")

    return parsed.model_dump()


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
