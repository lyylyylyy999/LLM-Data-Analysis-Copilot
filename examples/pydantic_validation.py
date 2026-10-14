"""学习任务 06B：只验证本地数据，不调用大模型接口。"""

import json

from pydantic import ValidationError

from data_copilot.schemas import AnalysisAnswer

VALID_ANSWER = {
    "summary": "  过拟合是模型过度适应训练数据。  ",
    "explanation": "训练表现好，不代表对未见数据的预测也好。",
    "missing_information": [],
}

INSUFFICIENT_DATA_ANSWER = {
    "summary": "没有配对观测数据，无法计算相关系数。",
    "explanation": "需要每个样本对应的广告投入和销售额。",
    "missing_information": ["广告投入与销售额的成对观测值"],
}

INVALID_ANSWERS = {
    "非法 JSON": '{"summary":',
    "缺少字段": '{"summary":"回答","explanation":"解释"}',
    "额外字段": (
        '{"summary":"回答","explanation":"解释",'
        '"missing_information":[],"confidence":0.9}'
    ),
    "字段类型错误": ('{"summary":123,"explanation":"解释","missing_information":[]}'),
    "列表元素类型错误": (
        '{"summary":"回答","explanation":"解释","missing_information":[42]}'
    ),
    "空摘要": ('{"summary":"   ","explanation":"解释","missing_information":[]}'),
}


def main() -> None:
    # 1. model_validate()：Python dict -> Pydantic 模型
    from_dict = AnalysisAnswer.model_validate(VALID_ANSWER)
    print("[model_validate]", from_dict.model_dump())
    assert from_dict.summary == VALID_ANSWER["summary"]

    # 2. model_validate_json()：JSON 字符串 -> Pydantic 模型
    raw_json = json.dumps(INSUFFICIENT_DATA_ANSWER, ensure_ascii=False)
    from_json = AnalysisAnswer.model_validate_json(raw_json)
    print("[model_validate_json]", from_json.model_dump())

    # 3. model_dump() / model_dump_json()：导出数据
    print("[model_dump]", from_dict.model_dump())
    print("[model_dump_json]", from_dict.model_dump_json())
    json.dumps(from_dict.model_dump(), ensure_ascii=False, allow_nan=False)

    # 4. model_json_schema()：查看自动生成的 JSON Schema
    schema = AnalysisAnswer.model_json_schema()
    print("[model_json_schema]")
    print(json.dumps(schema, ensure_ascii=False, indent=2, allow_nan=False))

    # 5. 不合格 JSON：直接显示 ValidationError，不自动修补
    for case_name, invalid_json in INVALID_ANSWERS.items():
        try:
            AnalysisAnswer.model_validate_json(invalid_json)
        except ValidationError as exc:
            print(f"[{case_name}] 拒绝：{exc.errors()[0]['type']}")
        else:
            raise AssertionError(f"无效数据未被拒绝：{case_name}")


if __name__ == "__main__":
    main()
