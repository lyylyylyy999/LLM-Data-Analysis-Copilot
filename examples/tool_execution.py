"""顺序执行同一轮的多个数据分析工具，将真实结果交回模型生成中文回答。

从项目根目录运行：
    python -m examples.tool_execution --csv data/samples/sales.csv
    python -m examples.tool_execution --csv data/samples/sales.csv \
        --question "请描述 sales 的基本统计特征。"
    python -m examples.tool_execution --csv data/samples/sales.csv \
        --question "sales 与 advertising_spend 的 Pearson 相关系数是多少？"

任务 08 的 examples/tool_calling.py 只负责观察工具请求，不修改它。
"""

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from data_copilot.analysis_tools import (
    correlation,
    dataset_info,
    describe_numeric,
    group_aggregate,
)
from data_copilot.data_loader import load_csv
from examples.tool_calling import QUESTION
from examples.tool_calling import TOOLS as TASK08_TOOLS

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageFunctionToolCall

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
    {
        "type": "function",
        "function": {
            "name": "group_aggregate",
            "description": (
                "按指定字段分组，对数值字段执行 mean、sum、count、min 或 max。"
                "count 仅统计目标列的非缺失值；缺失的分组键保留为 null；"
                "目标值全部缺失的组，其 sum 为 null 而非 0。"
                "数据由 Python 管理，无需传入 DataFrame。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group_column": {
                        "type": "string",
                        "description": "用于分组的列名，如 region 或 category。",
                    },
                    "target_column": {
                        "type": "string",
                        "description": "需要聚合的数值列名，如 sales。",
                    },
                    "aggregation": {
                        "type": "string",
                        "enum": ["mean", "sum", "count", "min", "max"],
                        "description": (
                            "聚合方式：平均值、总和、非缺失计数、最小值或最大值。"
                        ),
                    },
                },
                "required": ["group_column", "target_column", "aggregation"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "correlation",
            "description": (
                "计算两个数值字段之间的 Pearson 相关系数，仅使用两列同时非缺失"
                "的样本对。无法计算时系数为 null，并以 status 说明原因。"
                "相关关系不代表因果，工具不提供 p 值或置信区间。"
                "数据集由 Python 管理，无需传入 DataFrame。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x_column": {
                        "type": "string",
                        "description": "第一个数值列名，如 sales。",
                    },
                    "y_column": {
                        "type": "string",
                        "description": "第二个数值列名，如 advertising_spend。",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["pearson"],
                        "description": "相关系数的计算方法，目前仅支持 pearson。",
                    },
                },
                "required": ["x_column", "y_column", "method"],
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
    "询问按地区、商品类别等分组的聚合结果时使用 group_aggregate，"
    "准确填写 group_column、target_column 和 aggregation。"
    "平均值选 mean，总和选 sum，非缺失目标值数量选 count，"
    "最小值选 min，最大值选 max。"
    "实际执行工具之前不得猜测统计结果。"
    "只依据 Python 工具返回的结果回答：描述统计的 count 是非缺失值数量，"
    "缺失值不参与统计；标准差采用 ddof=1（样本标准差）。"
    "如未提供总行数，不要根据 count 推断缺失值的具体数量。"
    "group_aggregate 的 count 仅统计目标列的非缺失值，不是组内总行数；"
    "缺失分组键会以 null 保留，不能忽略；全缺失组的 sum 为 null，不能当作 0。"
    "询问两个数值字段之间的 Pearson 相关性时使用 correlation，"
    "提供 x_column、y_column、method=pearson。"
    "相关工具的 sample_size 是两列同时非缺失的完整观测对数，"
    "不等于任意单列的非缺失数量；不插补缺失值。"
    "当 status 为 insufficient_samples、zero_variance 或其他无法计算状态时，"
    "coefficient=null 代表相关系数无法计算，绝不代表相关系数等于零。"
    "相关关系不等于因果关系，不能据此证明广告导致销售额提高。"
    "该工具未返回 p 值或置信区间，不得声称已完成显著性检验。"
    "不将描述统计或分组差异解释为因果关系。"
    "复合问题必须在第一轮一次性请求全部必要工具；"
    "同一工具也可按不同参数多次请求，不要留待后续轮次再请求。"
    "收到工具执行结果后，直接根据已有结果回答，不要再次请求工具。"
)


def execute_tool(
    dataframe: pd.DataFrame, tool_name: str, arguments_json: str
) -> dict[str, Any]:
    """白名单分发四种工具；严格验证参数后交由真实统计函数执行。"""
    if tool_name not in {
        "dataset_info",
        "describe_numeric",
        "group_aggregate",
        "correlation",
    }:
        raise ValueError(f"不支持的工具名称：{tool_name}")

    # 无效 JSON 原样抛出 JSONDecodeError；数组、null 等非对象也应拒绝。
    arguments = json.loads(arguments_json)
    if not isinstance(arguments, dict):
        raise ValueError("工具调用 arguments 必须是 JSON 对象")

    if tool_name == "dataset_info":
        if arguments:
            raise ValueError(f"dataset_info 不接受参数：{arguments}")
        return dataset_info(dataframe)

    if tool_name == "describe_numeric":
        if set(arguments) != {"columns"}:
            raise ValueError("describe_numeric 必须且只能包含 columns 参数")
        columns = arguments["columns"]
        if not isinstance(columns, list):
            raise ValueError("columns 必须是字符串数组，不能是单个字符串")
        if not columns:
            raise ValueError("columns 至少包含一个列名")
        if any(not isinstance(column, str) for column in columns):
            raise ValueError("columns 中的每个列名都必须是字符串")

        # 复用统计函数对空白、重复、未知及非数值列的校验。
        return describe_numeric(dataframe, cast(list[str], columns))

    if tool_name == "group_aggregate":
        required = {"group_column", "target_column", "aggregation"}
        if set(arguments) != required:
            raise ValueError("group_aggregate 必须且只能包含三个必填参数")
        if any(not isinstance(arguments[name], str) for name in required):
            raise ValueError("group_aggregate 的三个参数都必须是字符串")

        # 现有统计函数校验列是否存在、目标类型与聚合方法的允许值。
        return group_aggregate(
            dataframe,
            group_column=arguments["group_column"],
            target_column=arguments["target_column"],
            aggregation=arguments["aggregation"],
        )

    required = {"x_column", "y_column", "method"}
    if set(arguments) != required:
        raise ValueError("correlation 必须且只能包含三个必填参数")
    if any(not isinstance(arguments[name], str) for name in required):
        raise ValueError("correlation 的三个参数都必须是字符串")

    # 底层函数检查列名、数值类型及 method == "pearson"。
    # 仅允许明确的调用分支，不执行模型提供的任意函数名称。
    return correlation(
        dataframe,
        x_column=arguments["x_column"],
        y_column=arguments["y_column"],
        method=arguments["method"],
    )


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

    # 第一轮允许模型一次返回多个独立工具请求；Python 仍按顺序执行。
    # 只提供字段元数据及工具 Schema，不提前提供统计数据。
    first_response = client.chat.completions.create(
        model=model,
        messages=messages.copy(),
        tools=TOOLS,
        tool_choice="auto",
        parallel_tool_calls=True,
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
    # 先统一检查整批调用的类型和 ID，任何工具执行之前就拒绝不合法的批次。
    seen_ids: set[str] = set()
    function_calls: list[ChatCompletionMessageFunctionToolCall] = []
    for tool_call in tool_calls:
        if tool_call.type != "function":
            raise ValueError(f"不支持的工具调用类型：{tool_call.type}")
        if not isinstance(tool_call.id, str) or not tool_call.id.strip():
            raise ValueError("工具调用 ID 必须是非空字符串")
        if tool_call.id in seen_ids:
            raise ValueError(f"工具调用 ID 重复：{tool_call.id}")
        seen_ids.add(tool_call.id)
        # 已检查 type == "function"，在此收窄 SDK 的联合类型。
        function_calls.append(cast("ChatCompletionMessageFunctionToolCall", tool_call))

    print(f"第一轮工具调用数量：{len(function_calls)}")
    print("第一轮工具调用请求：")
    print(
        json.dumps(
            [call.model_dump(exclude_none=True) for call in tool_calls],
            ensure_ascii=False,
        )
    )

    # assistant 消息必须包含整批工具调用，而不是拆成多条 assistant 消息。
    assistant_message: ChatCompletionMessageParam = {
        "role": "assistant",
        "content": first_message.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in function_calls
        ],
    }

    tool_messages: list[ChatCompletionMessageParam] = []
    for index, tool_call in enumerate(function_calls, start=1):
        print(
            f"工具调用 {index}：ID={tool_call.id}，"
            f"名称={tool_call.function.name}，"
            f"arguments={tool_call.function.arguments}"
        )
        # 参数校验、工具执行和 JSON 序列化中的异常直接传播。
        # 不伪造结果，也不会发送第二轮请求。
        result = execute_tool(
            dataframe, tool_call.function.name, tool_call.function.arguments
        )
        result_json = json.dumps(result, ensure_ascii=False, allow_nan=False)
        print(f"Python 实际执行的工具结果（{tool_call.id}）：")
        print(result_json)

        tool_message: ChatCompletionMessageParam = {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result_json,
        }
        tool_messages.append(tool_message)
        print("回传的 tool 消息：")
        print(json.dumps(tool_message, ensure_ascii=False))

    # 全部调用成功后，按模型请求顺序回传每一条工具结果。
    messages.append(assistant_message)
    messages.extend(tool_messages)

    # 第二轮显式禁止工具调用；仅允许根据已回传的结果生成正文。
    second_response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        tool_choice="none",
        extra_body={"enable_thinking": False},
        stream=False,
    )
    if not second_response.choices:
        raise RuntimeError("第二轮模型未返回任何 choice")

    second_choice = second_response.choices[0]
    print(f"第二轮结束原因：{second_choice.finish_reason}")
    if second_choice.finish_reason != "stop":
        if second_choice.message.tool_calls:
            print("第二轮意外返回的工具调用：")
            print(
                json.dumps(
                    [
                        call.model_dump(exclude_none=True)
                        for call in second_choice.message.tool_calls
                    ],
                    ensure_ascii=False,
                )
            )
        if second_choice.message.content:
            print(f"第二轮异常正文：{second_choice.message.content}")
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
