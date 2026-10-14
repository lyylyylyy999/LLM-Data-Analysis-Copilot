import os

from dotenv import load_dotenv
from openai import OpenAI


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
        messages=[{"role": "user", "content": question}],
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
