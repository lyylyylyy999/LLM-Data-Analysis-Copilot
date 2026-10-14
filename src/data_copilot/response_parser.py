import json


def parse_analysis_answer(answer: str) -> dict:
    """解析模型原始正文；JSON 语法错误直接保留 JSONDecodeError。"""
    parsed = json.loads(answer)

    if not isinstance(parsed, dict):
        raise ValueError("模型回答必须是 JSON 对象")

    required_fields = {
        "summary",
        "explanation",
        "missing_information",
    }
    actual_fields = set(parsed)

    missing_fields = required_fields - actual_fields
    extra_fields = actual_fields - required_fields

    if missing_fields:
        raise ValueError(f"缺少必需字段：{', '.join(sorted(missing_fields))}")

    if extra_fields:
        raise ValueError(f"存在额外字段：{', '.join(sorted(extra_fields))}")

    if not isinstance(parsed["summary"], str):
        raise ValueError("summary 必须是字符串")

    if not parsed["summary"].strip():
        raise ValueError("summary 不能为空或仅包含空白字符")

    if not isinstance(parsed["explanation"], str):
        raise ValueError("explanation 必须是字符串")

    missing_information = parsed["missing_information"]

    if not isinstance(missing_information, list):
        raise ValueError("missing_information 必须是字符串列表")

    if not all(isinstance(item, str) for item in missing_information):
        raise ValueError("missing_information 的每个元素都必须是字符串")

    return parsed
