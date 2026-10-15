"""用 Python 计算销售额统计量，再让模型解释结果。"""

import json
from pathlib import Path

from data_copilot.analysis_tools import dataset_info, describe_numeric
from data_copilot.data_loader import load_csv
from data_copilot.llm_client import ask_llm
from data_copilot.response_parser import parse_analysis_answer

QUESTION = (
    "这份数据有多少条记录？销售额缺失多少条？平均销售额是多少？请说明缺失值的处理方式。"
)


def main() -> None:
    csv_path = Path(__file__).resolve().parents[1] / "data" / "samples" / "sales.csv"
    dataframe = load_csv(csv_path)

    info = dataset_info(dataframe)
    numeric = describe_numeric(dataframe, "sales")
    sales = numeric["statistics"]["sales"]

    # 仅向模型提供本次问题需要的统计量，不发送 DataFrame 或原始记录。
    statistics = {
        "row_count": info["row_count"],
        "missing_values": {"sales": info["missing_values"]["sales"]},
        "statistics": {
            "sales": {
                "count": sales["count"],
                "mean": sales["mean"],
            }
        },
    }
    statistics_json = json.dumps(statistics, ensure_ascii=False, allow_nan=False)

    prompt = (
        f"问题：{QUESTION}\n\n"
        "以下统计 JSON 是 Python 工具从 CSV 中计算得到的结果，"
        "不是你读取 CSV 或执行统计计算得到的。回答中的所有数值必须以它为依据。\n"
        "row_count 表示数据总记录数；missing_values.sales 表示销售额缺失记录数；"
        "statistics.sales.count 表示销售额非缺失记录数。"
        "请解释总记录数与有效记录数的区别。\n"
        "statistics.sales.mean 是仅使用非缺失销售额计算出的均值；"
        "缺失销售额被跳过，没有补零。均值可合理舍入，但不能改变原有统计含义。\n"
        "如果 mean 为 null，请说明均值不可用，不要编造数值。"
        "不得添加统计 JSON 无法支持的业务结论，"
        "也不得声称自己读取过 CSV 或计算过统计量。\n"
        "只输出合法 JSON 对象，且仅包含 "
        "summary、explanation、missing_information 三个字段；"
        "summary 和 explanation 为字符串，missing_information 为字符串列表。\n\n"
        f"统计 JSON：\n{statistics_json}"
    )

    result = ask_llm(prompt)
    answer = parse_analysis_answer(result["answer"])

    # 全部步骤成功后再展示；失败时保留原始异常和 traceback。
    print("Python 统计结果：")
    print(json.dumps(statistics, ensure_ascii=False, allow_nan=False, indent=2))
    print("模型解释：")
    print(json.dumps(answer, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
