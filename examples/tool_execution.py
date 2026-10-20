"""加载销售 CSV 并展示可复用分析服务返回的工具结果和最终回答。

运行示例：
    python -m examples.tool_execution --csv data/samples/sales.csv
    python -m examples.tool_execution --csv data/samples/sales.csv \
        --question "每个地区的平均销售额是多少？"
"""

import argparse
import json
from pathlib import Path
from typing import Any

from data_copilot.analysis_service import analyze_question
from data_copilot.data_loader import load_csv

QUESTION = "这份数据有多少行、多少列？"


def main(csv_path: Path, question: str = QUESTION) -> dict[str, Any]:
    """CLI 适配层：读取 CSV，调用服务并展示结果。"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("问题不能为空")
    dataframe = load_csv(csv_path)
    result = analyze_question(dataframe, question)
    tool_results = result["tool_results"]
    if not tool_results:
        print("本次没有请求工具；不执行统计，也不发送第二次请求。")
    for index, item in enumerate(tool_results, start=1):
        print(
            f"工具调用 {index}：ID={item['tool_call_id']}，"
            f"名称={item['name']}，"
            f"arguments={json.dumps(item['arguments'], ensure_ascii=False)}"
        )
        print("Python 实际执行的工具结果：")
        print(json.dumps(item["result"], ensure_ascii=False, allow_nan=False))
    print(f"最终模型：{result['model']}")
    print(f"结束原因：{result['finish_reason']}")
    print(f"模型最终回答：{result['answer']}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="销售 CSV 文件路径")
    parser.add_argument("--question", default=QUESTION, help="发送给模型的问题")
    args = parser.parse_args()
    main(args.csv, args.question)
