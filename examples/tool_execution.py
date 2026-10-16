"""执行一次 dataset_info 工具调用，并将 Python 的真实统计结果交回模型。

从项目根目录运行：
    python -m examples.tool_execution --csv data/samples/sales.csv

原任务 08 的 examples/tool_calling.py 只负责观察工具请求，此文件不修改它。
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from data_copilot.analysis_tools import dataset_info
from data_copilot.data_loader import load_csv
from examples.tool_calling import QUESTION, TOOLS

SYSTEM_PROMPT = (
    "当前数据集由 Python 管理，具体统计信息尚未提供。"
    "需要行列数时可以请求 dataset_info 工具；"
    "工具真正执行之前，不要猜测数据内容。"
)


def main(csv_path: Path) -> None:
    # 数据只由 Python 加载、持有；第一轮请求不会包含数据或统计结果。
    dataframe = load_csv(csv_path)

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
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": QUESTION},
    ]

    # 第一次请求：沿用任务 08 的工具声明和自动工具选择。
    first_response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        extra_body={"enable_thinking": False},
        stream=False,
    )
    if not first_response.choices:
        raise RuntimeError("第一轮模型未返回任何 choice")

    first_choice = first_response.choices[0]
    first_message = first_choice.message
    print(f"第一轮模型：{first_response.model}")
    print(f"第一轮结束原因：{first_choice.finish_reason}")

    if first_choice.finish_reason == "length":
        raise RuntimeError("第一轮模型响应被截断")
    if first_message.refusal:
        raise RuntimeError(f"第一轮模型拒绝回答：{first_message.refusal}")
    if first_choice.finish_reason not in {"stop", "tool_calls"}:
        raise RuntimeError(f"第一轮模型状态异常：{first_choice.finish_reason}")

    tool_calls = first_message.tool_calls
    if not tool_calls:
        if first_choice.finish_reason != "stop":
            raise RuntimeError("第一轮标记为 tool_calls，但未返回工具调用")
        print("本次没有请求工具；不执行统计，也不发送第二次请求。")
        if first_message.content:
            print(f"模型正文：{first_message.content}")
        return

    if first_choice.finish_reason != "tool_calls":
        raise RuntimeError("第一轮返回了工具调用，但结束原因不是 tool_calls")
    if len(tool_calls) != 1:
        raise ValueError(
            f"本任务必须恰好调用一次 dataset_info，实际为 {len(tool_calls)} 次"
        )

    tool_call = tool_calls[0]
    if tool_call.type != "function":
        raise ValueError(f"不支持的工具调用类型：{tool_call.type}")
    if tool_call.function.name != "dataset_info":
        raise ValueError(f"不支持的工具名称：{tool_call.function.name}")
    if not isinstance(tool_call.id, str) or not tool_call.id.strip():
        raise ValueError("工具调用 ID 不能为空")

    # JSONDecodeError 原样抛出；数组、null 以及含任何额外字段的对象均拒绝。
    arguments = json.loads(tool_call.function.arguments)
    if not isinstance(arguments, dict):
        raise ValueError("工具调用 arguments 必须是 JSON 对象")
    if arguments != {}:
        raise ValueError(f"dataset_info 不接受额外参数：{arguments}")

    assistant_message: ChatCompletionMessageParam = {
        "role": "assistant",
        "content": first_message.content,
        "tool_calls": [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
        ],
    }
    print("第一轮工具调用请求：")
    print(json.dumps([tool_call.model_dump(exclude_none=True)], ensure_ascii=False))

    # 不依据模型参数提供 DataFrame；只调用明确允许的 Python 函数。
    result = dataset_info(dataframe)
    result_json = json.dumps(result, ensure_ascii=False, allow_nan=False)
    tool_message: ChatCompletionMessageParam = {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": result_json,
    }
    print("Python 实际执行的工具结果：")
    print(result_json)
    print("回传的 tool 消息：")
    print(json.dumps(tool_message, ensure_ascii=False))

    # system → user → assistant(完整 tool_calls) → tool(同一 ID)。
    messages.extend([assistant_message, tool_message])

    # 第二次请求不给 tools，也不加 JSON Schema；继续使用同一条消息链。
    second_response = client.chat.completions.create(
        model=model,
        messages=messages,
        extra_body={"enable_thinking": False},
        stream=False,
    )
    if not second_response.choices:
        raise RuntimeError("第二轮模型未返回任何 choice")

    second_choice = second_response.choices[0]
    print(f"第二轮结束原因：{second_choice.finish_reason}")
    if second_choice.finish_reason != "stop":
        raise RuntimeError(f"第二轮模型未正常结束：{second_choice.finish_reason}")
    if second_choice.message.refusal:
        raise RuntimeError(f"第二轮模型拒绝回答：{second_choice.message.refusal}")
    if second_choice.message.tool_calls:
        raise RuntimeError("第二轮意外返回了工具调用")
    answer = second_choice.message.content
    if not answer or not answer.strip():
        raise RuntimeError("第二轮模型未返回有效正文")
    print(f"模型最终回答：{answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="模拟销售 CSV 文件路径")
    args = parser.parse_args()
    main(args.csv)
