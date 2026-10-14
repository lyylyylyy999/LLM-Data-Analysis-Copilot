import os
import sys

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError


def main() -> None:
    """调用大模型 API，并流式输出思考过程与正式回复。"""

    # 1. 加载环境变量
    load_dotenv()

    api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    base_url = (os.getenv("DASHSCOPE_BASE_URL") or "").strip()
    model = (os.getenv("DASHSCOPE_MODEL") or "").strip()

    # 2. 验证配置
    required_config = {
        "DASHSCOPE_API_KEY": api_key,
        "DASHSCOPE_BASE_URL": base_url,
        "DASHSCOPE_MODEL": model,
    }

    missing = [
        name
        for name, value in required_config.items()
        if not value or not value.strip()
    ]

    if missing:
        raise RuntimeError(f"缺少必要的环境变量：{', '.join(missing)}")

    # 3. 创建 API 客户端
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=60.0,
    )

    # 4. 构造对话消息
    messages = [
        {
            "role": "user",
            "content": "简单介绍一下你自己",
        }
    ]

    # 5. 初始化状态
    is_answering = False
    answer_parts = []
    model_name = None
    usage = None
    finish_reason = None

    try:
        # 6. 发送流式请求
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body={"enable_thinking": True},
            stream=True,
            stream_options={"include_usage": True},
        )

        print("\n" + "=" * 20 + "思考过程" + "=" * 20)

        # 7. 逐块处理流式响应
        for chunk in completion:
            # 记录实际返回的模型名称
            if model_name is None:
                model_name = getattr(chunk, "model", None)

            # 记录 Token 用量
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = chunk_usage

            # 部分数据块可能只包含 usage
            if not chunk.choices:
                continue

            choice = chunk.choices[0]

            # 记录生成结束原因
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason

            delta = choice.delta

            # 8. 输出思考内容
            reasoning = getattr(delta, "reasoning_content", None)

            if reasoning and not is_answering:
                print(reasoning, end="", flush=True)

            # 9. 输出正式回复
            if delta.content:
                if not is_answering:
                    print("\n" + "=" * 20 + "回复内容（流式）" + "=" * 20)
                    is_answering = True

                answer_parts.append(delta.content)
                print(delta.content, end="", flush=True)

        # 10. 检查生成结束原因
        if finish_reason == "length":
            raise RuntimeError("回复被截断：模型达到生成长度限制")

        if finish_reason is None:
            raise RuntimeError("未收到生成结束标记，无法确认回复完整")

        if finish_reason != "stop":
            raise RuntimeError(f"模型未正常完成生成：{finish_reason}")

        # 11. 检查是否存在有效正文
        if not is_answering or not "".join(answer_parts).strip():
            raise RuntimeError("模型未返回正文")

        # 12. 输出调用信息
        print("\n\n" + "=" * 20 + "调用信息" + "=" * 20)

        print("配置模型：", model)
        print("实际返回模型：", model_name or "未返回")
        print("结束原因：", finish_reason)

        if usage is not None:
            print("输入 tokens：", usage.prompt_tokens)
            print("输出 tokens：", usage.completion_tokens)
            print("总 tokens：", usage.total_tokens)
        else:
            print("未获取到 Token 用量")

    except OpenAIError as exc:
        print(
            f"\nAPI 调用失败：{exc}",
            file=sys.stderr,
        )
        raise


if __name__ == "__main__":
    main()
