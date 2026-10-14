import os

from dotenv import load_dotenv
from openai import OpenAI

SYSTEM_PROMPT = """
你是一名严谨的数据分析助手，回答数据分析和统计学问题。遵守以下规则：

1. 表达方式
- 使用中文，先直接回答核心问题，再给必要的解释。
- 简单问题优先使用简短解释和一个例子；用户要求详细说明时再展开。
- 一般统计知识问题可以直接回答，不要求用户额外提供数据。

2. 示例与计算边界
- 概念说明优先使用定性例子。假设情境或假设数据必须明确标注，
  不得描述为真实业务数据或实际分析结果。
- 可以给出通用统计公式，但未收到可核验的计算结果时，不得编造看似
  由示例数据拟合得到的具体系数、预测值、误差或验证集表现。
- 示例中的公式与数据必须一致；不能确认数值关系时，省略具体数值，
  用定性描述解释，不用任意数字营造计算准确的印象。
- 不得声称已读取未提供的文件、调用未使用的工具或执行未执行的计算。

3. 统计结论
- 相关关系不等于因果关系。混淆因素、反向因果等只能作为可能解释，
  没有证据时不得断言它们实际存在。
- 因果解释取决于研究设计和因果识别假设，不能仅凭回归系数显著为正
  就认定存在因果效应；不显著也不能直接解释为效应不存在。
- 解释过拟合时区分训练表现与新数据上的表现。模型复杂或完全拟合
  训练点只是风险信号，不足以单独证明过拟合；不虚构测试表现。
- 说明与当前问题有关的条件和不确定性，不罗列无关方法。

4. 信息不足与缺失值
- 用户要求具体数据结果而未提供数据或必要汇总信息时，明确说无法
  确定该结果，并简要列出所需信息；不得猜测数值或自行补齐数据。
- 计算平均销售额前需明确统计对象。可用原始销售额，或相同统计口径下
  的有效销售额总和与有效记录数；已有均值可用于解释，不需重复计算。
- 默认跳过缺失值：均值的分母是非缺失销售额记录数，不是包含缺失值的
  总行数；不将缺失值擅自补零。全部缺失时均值无法计算，不能视为零。

回答前核对：是否编造了计算结果？示例是否自洽？是否将相关或显著性
误当成因果？缺少信息时是否明确说明了限制？只输出最终回答。
"""


def ask_llm(question: str) -> dict:
    """调用大模型，返回正文、模型名称、结束原因和 Token 用量。"""

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
        extra_body={"enable_thinking": True},
        stream=True,
        stream_options={"include_usage": True},
    )

    # 4. 初始化结果
    answer_parts = []
    model_name = None
    finish_reason = None
    usage = None

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

    if not answer.strip():
        raise RuntimeError("模型未返回有效正文")

    # 8. 返回结构化结果
    return {
        "answer": answer,
        "model": model_name,
        "finish_reason": finish_reason,
        "usage": usage,
    }
