"""使用 Phase 0 工具分析模拟数据或用户指定的 CSV。"""

import argparse
import json
from pathlib import Path

from data_copilot.analysis_tools import (
    correlation,
    dataset_info,
    describe_numeric,
    group_aggregate,
)
from data_copilot.data_loader import dataset_overview, load_csv


def main() -> None:
    """输出 JSON 报告；CSV 必须包含 region、sales 和 advertising_spend 列。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/samples/sales.csv",
        help="CSV path (default: repository sales sample)",
    )
    parser.add_argument("--encoding", default="utf-8-sig")
    args = parser.parse_args()
    dataframe = load_csv(args.csv_path, encoding=args.encoding)
    results = {
        "info": dataset_info(dataframe),
        "overview": dataset_overview(dataframe, preview_rows=3),
        "description": describe_numeric(dataframe, ["sales", "advertising_spend"]),
        "by_region": group_aggregate(dataframe, "region", "sales"),
        "correlation": correlation(dataframe, "sales", "advertising_spend"),
    }
    print(json.dumps(results, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
