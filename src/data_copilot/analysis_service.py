"""可复用的数据分析服务：有限轮工具调用、真实结果回传和最终回答。"""

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
    "初次请求仅提供列名及数据类型，之后可根据工具结果继续分析。"
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
    "根据当前已知信息请求完成问题所需的工具：可以同轮请求多个工具，"
    "也可以在收到工具结果后，再请求其他工具完成依赖性分析。"
    "例如可先使用 dataset_info 识别缺失列，再使用 describe_numeric 统计它们。"
    "已有足够真实结果时直接用中文回答，不要重复请求已经完成的计算。"
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


def analyze_question(
    dataframe: pd.DataFrame,
    question: str,
    *,
    max_tool_rounds: int = 3,
) -> dict[str, Any]:
    """分析现有 DataFrame，最多执行 max_tool_rounds 批工具并生成最终回答。

    一批可包含多次工具调用；每批执行完后再次向模型请求下一步。
    已用满工具轮数时仍允许一次模型请求产生最终文本，但不再执行工具。
    """
    if isinstance(max_tool_rounds, bool) or not isinstance(max_tool_rounds, int):
        raise ValueError("max_tool_rounds 必须是正整数")
    if max_tool_rounds <= 0:
        raise ValueError("max_tool_rounds 必须是正整数")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("问题不能为空")
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe 必须是 pandas DataFrame")

    # 导入模块不读取配置或创建客户端；只在函数调用时读取。
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
    tool_results: list[dict[str, Any]] = []
    completed_rounds = 0

    # 最多 max_tool_rounds 批执行 + 最后一次仅用于回答或检测超限的模型请求。
    for _ in range(max_tool_rounds + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages.copy(),
            tools=TOOLS,
            tool_choice="auto",
            parallel_tool_calls=True,
            extra_body={"enable_thinking": False},
            stream=False,
        )
        round_name = f"第{completed_rounds + 1}次请求"
        if not response.choices:
            raise RuntimeError(f"{round_name}模型未返回任何 choice")

        choice = response.choices[0]
        message = choice.message
        if choice.finish_reason == "length":
            raise RuntimeError(f"{round_name}模型响应被截断")
        if message.refusal:
            raise RuntimeError(f"{round_name}模型拒绝回答：{message.refusal}")
        if choice.finish_reason not in {"stop", "tool_calls"}:
            raise RuntimeError(f"{round_name}模型状态异常：{choice.finish_reason}")

        tool_calls = message.tool_calls
        if not tool_calls:
            if choice.finish_reason != "stop":
                raise RuntimeError(f"{round_name}标记为 tool_calls，但未返回工具调用")
            answer = _validate_final_answer(choice, round_name)
            return _result_payload(answer, response.model, tool_results)

        if choice.finish_reason != "tool_calls":
            raise RuntimeError(
                f"{round_name}意外返回了工具调用（结束原因不是 tool_calls）"
            )
        if completed_rounds >= max_tool_rounds:
            raise RuntimeError(
                f"工具调用轮数超过上限 {max_tool_rounds}；"
                "模型继续请求工具，超限批次未执行"
            )

        # 执行本批任一工具之前，统一检查调用类型、ID 非空和批内 ID 唯一。
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
            function_calls.append(
                cast("ChatCompletionMessageFunctionToolCall", tool_call)
            )

        assistant_message: ChatCompletionMessageParam = {
            "role": "assistant",
            "content": message.content,
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
        batch_results: list[dict[str, Any]] = []
        for call in function_calls:
            # 不通过动态函数查找执行工具；底层校验与异常保持原样。
            tool_result = execute_tool(
                dataframe, call.function.name, call.function.arguments
            )
            result_json = json.dumps(tool_result, ensure_ascii=False, allow_nan=False)
            tool_messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result_json}
            )
            batch_results.append(
                {
                    "tool_call_id": call.id,
                    "name": call.function.name,
                    "arguments": json.loads(call.function.arguments),
                    "result": tool_result,
                }
            )

        # 执行成功后才追加完整 assistant 和所有 tool 消息；下一轮携带全部历史。
        messages.append(assistant_message)
        messages.extend(tool_messages)
        tool_results.extend(batch_results)
        completed_rounds += 1

    # for 循环本身有限；正常完成必须在循环中以 stop 返回。
    raise RuntimeError("工具调用轮数已达上限，模型未返回最终回答")
