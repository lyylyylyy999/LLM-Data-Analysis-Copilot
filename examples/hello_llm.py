import json

from data_copilot.llm_client import ask_llm


def main() -> None:
    question = "请告诉我这份数据的平均销售额。"

    result = ask_llm(question)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
