"""选择并执行一次数据分析工具，将真实结果交回模型生成中文回答。

从项目根目录运行：
    python -m examples.tool_execution --csv data/samples/sales.csv
    python -m examples.tool_execution --csv data/samples/sales.csv \
        --question "请描述 sales 的基本统计特征。"

任务 08 的 examples/tool_calling.py 只负责观察工具请求，不修改它。
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, cast

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from data_copilot.analysis_tools import dataset_info, describe_numeric
from data_copilot.data_loader import load_csv
from examples.tool_calling import QUESTION
from examples.tool_calling import TOOLS as TASK08_TOOLS

TOOLS: list[ChatCompletionToolParam] = [
    *TASK08_TOOLS,
    {
        "type": "function",
        "function": {
            "name": "describe_numeric",
            "description": (
                "计算指定数值列的描述统计，包括非缺失值数量、均值、标准差、"
                "分位数及极值。缺失值不参与计算，标准差使用 ddof=1。"
                "数据集由 Python 管理，无需传入数据。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "uniqueItems": True,
                        "description": "要分析的数值列名，必须来自当前数据集。",
                    }
                },
                "required": ["columns"],
                "additionalProperties": False,
            },
        },
    },
]

SYSTEM_PROMPT = (
    "你是严谨的数据分析助手，请用中文回答。当前数据集由 Python 管理。"
    "你只知道下方列名及数据类型，不知道数据值或统计结果。"
    "询问行数或列数时使用 dataset_info；询问数值列的描述统计时"
    "使用 describe_numeric，并将指定列名放进 columns 数组。"
    "实际执行工具之前不得猜测统计结果。"
    "只依据 Python 工具返回的结果回答：描述统计的 count 是非缺失值数量，"
    "缺失值不参与统计；标准差采用 ddof=1（样本标准差）。"
    "如未提供总行数，不要根据 count 推断缺失值的具体数量。"
    "不将描述统计或相关关系解释为因果关系。"
)


def execute_tool(
    dataframe: pd.DataFrame, tool_name: str, arguments_json: str
) -> dict[str, Any]:
    """只分发两种白名单工具；不信任模型提供的 JSON 参数。"""
    if tool_name not in {"dataset_info", "describe_numeric"}:
        raise ValueError(f"不支持的工具名称：{tool_name}")

    # 非法 JSON 直接保留 JSONDecodeError；JSON 数组、null 等均拒绝。
    arguments = json.loads(arguments_json)
    if not isinstance(arguments, dict):
        raise ValueError("工具调用 arguments 必须是 JSON 对象")

    if tool_name == "dataset_info":
        if arguments:
            raise ValueError(f"dataset_info 不接受参数：{arguments}")
        return dataset_info(dataframe)

    if set(arguments) != {"columns"}:
        raise ValueError("describe_numeric 必须且只能包含 columns 参数")
    columns = arguments["columns"]
    if not isinstance(columns, list):
        raise ValueError("columns 必须是字符串数组，不能是单个字符串")
    if not columns:
        raise ValueError("columns 至少包含一个列名")
    if any(not isinstance(column, str) for column in columns):
        raise ValueError("columns 中的每个列名都必须是字符串")

    # 复用现有工具对空白名称、重复名称、未知列、非数值列的校验。
    return describe_numeric(dataframe, cast(list[str], columns))


def main(csv_path: Path, question: str = QUESTION) -> None:
    if not question.strip():
        raise ValueError("问题不能为空")

    # DataFrame 仅由当前 Python 进程加载并保留，不传给模型。
    dataframe = load_csv(csv_path)
    column_types = {str(name): str(dtype) for name, dtype in dataframe.dtypes.items()}

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
        {
            "role": "system",
            "content": (
                f"{SYSTEM_PROMPT}\n当前字段名称与类型："
                f"{json.dumps(column_types, ensure_ascii=False)}"
            ),
        },
        {"role": "user", "content": question},
    ]

    # 第一轮：只提供字段元数据及工具 Schema，不提供统计数据。
    first_response = client.chat.completions.create(
        model=model,
        messages=messages.copy(),
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
        raise ValueError(f"本任务只支持一次工具调用，实际为 {len(tool_calls)} 次")

    tool_call = tool_calls[0]
    if tool_call.type != "function":
        raise ValueError(f"不支持的工具调用类型：{tool_call.type}")
    if not isinstance(tool_call.id, str) or not tool_call.id.strip():
        raise ValueError("工具调用 ID 不能为空")

    print("第一轮工具调用请求：")
    print(json.dumps([tool_call.model_dump(exclude_none=True)], ensure_ascii=False))

    # 校验/执行期间出现的异常直接传播；不生成伪造的工具结果。
    result = execute_tool(
        dataframe, tool_call.function.name, tool_call.function.arguments
    )
    result_json = json.dumps(result, ensure_ascii=False, allow_nan=False)
    print("Python 实际执行的工具结果：")
    print(result_json)

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
    tool_message: ChatCompletionMessageParam = {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": result_json,
    }
    print("回传的 tool 消息：")
    print(json.dumps(tool_message, ensure_ascii=False))
    messages.extend([assistant_message, tool_message])

    # 第二轮：复用 system → user → assistant(tool_calls) → tool，不再提供工具。
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
    parser.add_argument("--csv", type=Path, required=True, help="销售 CSV 文件路径")
    parser.add_argument("--question", default=QUESTION, help="发送给模型的问题")
    args = parser.parse_args()
    main(args.csv, args.question)
