"""数据分析工具的公开 Schema 与严格参数校验、白名单分发。"""

import json
from typing import Any, cast

import pandas as pd
from openai.types.chat import ChatCompletionToolParam

from data_copilot.analysis_tools import (
    correlation,
    dataset_info,
    describe_numeric,
    group_aggregate,
)

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
    },
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
