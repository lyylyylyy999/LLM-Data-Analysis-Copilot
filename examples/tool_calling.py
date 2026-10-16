"""观察模型的 dataset_info 工具调用请求，不执行任何统计工具。"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam

QUESTION = "这份数据有多少行、多少列？"

TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "dataset_info",
            "description": (
                "获取当前数据集的行数、列数、字段名称及各字段缺失数量。"
                "数据集由 Python 程序管理，无需传入参数。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    }
]


def main() -> None:
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

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "当前数据集由 Python 管理，具体统计信息尚未提供。"
                    "需要行列数时可以请求 dataset_info 工具；"
                    "工具真正执行之前，不要猜测数据内容。"
                ),
            },
            {"role": "user", "content": QUESTION},
        ],
        tools=TOOLS,
        tool_choice="auto",
        extra_body={"enable_thinking": False},
        stream=False,
    )

    if not response.choices:
        raise RuntimeError("模型未返回任何 choice")

    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise RuntimeError("模型响应被截断")

    message = choice.message
    print(f"模型：{response.model}")
    print(f"结束原因：{choice.finish_reason}")

    if not message.tool_calls:
        print("本次没有请求工具")
        if message.content:
            print(f"模型正文：{message.content}")
        return

    for index, call in enumerate(message.tool_calls, start=1):
        if call.type != "function":
            raise ValueError(f"不支持的工具调用类型：{call.type}")

        print(f"工具调用 {index}：")
        print(f"  调用 ID：{call.id}")
        print(f"  工具名称：{call.function.name}")
        print(f"  原始 arguments：{call.function.arguments}")
        arguments = json.loads(call.function.arguments)
        if not isinstance(arguments, dict):
            raise ValueError("工具调用 arguments 必须是 JSON 对象")
        print(f"  解析后参数（dict）：{arguments}")

    print("以上仅是模型提出的工具调用请求；Python 尚未执行工具。")


if __name__ == "__main__":
    main()
