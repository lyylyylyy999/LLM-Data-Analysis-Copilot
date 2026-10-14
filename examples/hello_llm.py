import json

from data_copilot.llm_client import ask_llm
from data_copilot.response_parser import parse_analysis_answer


def main() -> None:
    question = "请简单介绍一下你自己"

    result = ask_llm(question)
    parsed = parse_analysis_answer(result["answer"])

    print(
        json.dumps(
            parsed,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
