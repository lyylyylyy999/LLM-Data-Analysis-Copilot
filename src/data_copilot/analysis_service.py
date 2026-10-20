"""封装单轮工具批量执行和两轮模型请求的分析服务。"""

import json
import os
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import Choice

from data_copilot.tool_registry import TOOLS, execute_tool

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageFunctionToolCall

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


def _validate_final_answer(choice: Choice, round_name: str) -> str:
    """校验无后续工具请求的最终回复，拒绝异常状态和空正文。"""
    if choice.finish_reason != "stop":
        raise RuntimeError(f"{round_name}模型未正常结束：{choice.finish_reason}")
    if choice.message.refusal:
        raise RuntimeError(f"{round_name}模型拒绝回答：{choice.message.refusal}")
    if choice.message.tool_calls:
        raise RuntimeError(f"{round_name}意外返回了工具调用")
    content = choice.message.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"{round_name}模型未返回有效正文")
    return content


def _result_payload(
    answer: str, model: str, tool_results: list[dict[str, Any]]
) -> dict[str, Any]:
    """只返回严格 JSON 兼容的结构，不修改工具的原始统计值。"""
    result: dict[str, Any] = {
        "answer": answer,
        "model": model,
        "finish_reason": "stop",
        "tool_results": tool_results,
    }
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    return result


def analyze_question(dataframe: pd.DataFrame, question: str) -> dict[str, Any]:
    """分析现成 DataFrame，最多请求模型两次，同轮工具依次执行。"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("问题不能为空")
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe 必须是 pandas DataFrame")

    # 模块导入时没有环境或网络副作用；每次调用时才读取配置、创建客户端。
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

    column_types = {str(name): str(dtype) for name, dtype in dataframe.dtypes.items()}
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
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
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
        answer = _validate_final_answer(first_choice, "第一轮")
        return _result_payload(answer, first_response.model, [])

    if first_choice.finish_reason != "tool_calls":
        raise RuntimeError("第一轮返回了工具调用，但结束原因不是 tool_calls")

    # 执行任何工具之前，先校验整批请求的类型、ID 和重复 ID。
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
        function_calls.append(cast("ChatCompletionMessageFunctionToolCall", tool_call))

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
    tool_results: list[dict[str, Any]] = []
    for call in function_calls:
        # 原始参数与异常保留：不使用 eval/exec，也不从模型名称动态查找函数。
        tool_result = execute_tool(
            dataframe, call.function.name, call.function.arguments
        )
        result_json = json.dumps(tool_result, ensure_ascii=False, allow_nan=False)
        tool_messages.append(
            {"role": "tool", "tool_call_id": call.id, "content": result_json}
        )
        tool_results.append(
            {
                "tool_call_id": call.id,
                "name": call.function.name,
                "arguments": json.loads(call.function.arguments),
                "result": tool_result,
            }
        )

    # 只在全部工具成功后发送第二次请求；同一工具可按不同参数多次调用。
    messages.append(assistant_message)
    messages.extend(tool_messages)
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
    answer = _validate_final_answer(second_response.choices[0], "第二轮")
    return _result_payload(answer, second_response.model, tool_results)
