import os

from dotenv import load_dotenv
from openai import OpenAI

from data_copilot.schemas import AnalysisAnswer

SYSTEM_PROMPT = """
你是一名专业、严谨的数据分析助手，负责回答数据分析、统计学及相关问题。

请严格遵守以下规则：

1. 语言与表达
- 始终使用中文回答。
- 解释应当清晰、准确、易于理解。
- 根据问题的复杂程度提供必要的解释。
- 对专业统计概念，可以结合公式或简单示例进行说明。

2. 数据真实性
- 不得编造数据、统计结果、计算结果或分析结论。
- 不得将假设性示例描述为真实的分析结果。
- 如果使用假设数据进行概念说明，必须明确标注为假设示例。

3. 统计解释
- 严格区分相关关系与因果关系。
- 不得仅根据相关性推断因果关系。
- 解释统计结果时，应说明结论成立所依赖的条件。
- 对无法确定的结论，应明确说明不确定性。

4. 信息不足时的处理
- 如果用户要求具体的数据分析，但没有提供数据或统计结果，
  必须明确说明当前缺少哪些必要信息。
- 不得自行假设缺失的数据并据此生成实际分析结论。
- 可以说明完成分析需要的数据、变量或统计指标。
- 如果问题属于一般统计知识解释，不需要用户额外提供数据，
  应直接回答。

5. 回答范围
- 仅根据用户提供的信息和可靠的统计学知识回答。
- 不得声称已经读取、处理或分析未提供的数据。
- 不得声称执行过实际未执行的统计计算。

6. JSON 输出格式（必须严格遵守）
- 只返回一个合法的 JSON 对象，不得添加 Markdown 代码围栏、前言、后记或 JSON 以外的文字。
- JSON 对象必须且只能包含以下三个字段，字段名称保持完全一致：
  "summary"、"explanation"、"missing_information"。
- "summary" 必须是非空字符串，用于直接回答核心问题。
- "explanation" 必须是字符串，用于提供必要的解释；可以为空字符串。
- "missing_information" 必须是字符串列表。
  若问题信息充分，或者是无需数据的一般知识问题，返回空列表 []。
- 如果缺少完成具体数据分析所必需的数据或统计结果，
  在 "missing_information" 中逐条说明缺少的信息，不要虚构数据。
- 字符串中的双引号、换行等字符必须正确转义，确保返回内容可被标准 JSON 解析器解析。

返回结构示例（仅展示字段和类型，实际回答应替换内容）：
{
  "summary": "直接回答核心问题",
  "explanation": "必要的解释",
  "missing_information": []
}
"""


def ask_llm(question: str) -> dict:
    """流式请求接口级结构化输出，校验后返回原始 JSON 正文及调用元数据。"""

    if not question.strip():
        raise ValueError("question 不能为空或空白内容")

    # 1. 加载并检查环境变量
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

    # 2. 创建客户端
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=60.0,
    )

    # 3. 发送流式请求
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "AnalysisAnswer",
                "strict": True,
                "schema": AnalysisAnswer.model_json_schema(),
            },
        },
        extra_body={"enable_thinking": False},
        stream=True,
        stream_options={"include_usage": True},
    )

    # 4. 初始化结果
    answer_parts = []
    model_name = None
    finish_reason = None
    usage = None
    refused = False

    # 5. 处理流式响应
    for chunk in response:
        if model_name is None:
            model_name = getattr(chunk, "model", None)

        # 保存 Token 用量
        chunk_usage = getattr(chunk, "usage", None)

        if chunk_usage is not None:
            usage = {
                "prompt_tokens": chunk_usage.prompt_tokens,
                "completion_tokens": chunk_usage.completion_tokens,
                "total_tokens": chunk_usage.total_tokens,
            }

        # 统计信息块可能没有 choices
        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        if getattr(choice.delta, "refusal", None):
            refused = True

        # 保存结束原因
        if choice.finish_reason is not None:
            finish_reason = choice.finish_reason

        # 只保存正文，忽略思考内容
        content = choice.delta.content

        if content:
            answer_parts.append(content)

    # 6. 拼接最终回答
    answer = "".join(answer_parts)

    # 7. 检查是否正常结束
    if finish_reason == "length":
        raise RuntimeError("模型回答被截断")

    if finish_reason is None:
        raise RuntimeError("未收到生成结束标记")

    if finish_reason != "stop":
        raise RuntimeError(f"模型未正常结束：{finish_reason}")

    if refused:
        raise RuntimeError("模型拒绝回答")

    if not answer.strip():
        raise RuntimeError("模型未返回有效正文")

    # 接口约束之外，仍需本地校验（例如 summary 不得纯空白）。
    AnalysisAnswer.model_validate_json(answer)

    # 8. 保留原有返回格式，answer 是通过校验的原始 JSON 字符串。
    return {
        "answer": answer,
        "model": model_name,
        "finish_reason": finish_reason,
        "usage": usage,
    }
