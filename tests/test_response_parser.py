"""Validate model JSON locally, without API calls or fabricated fallback data."""

import json

import pytest

from data_copilot.response_parser import parse_analysis_answer


@pytest.fixture
def valid_answer():
    return {
        "summary": "缺少数据，无法确定平均销售额。",
        "explanation": "请提供销售额总和和有效记录数。",
        "missing_information": ["销售额数据"],
    }


def test_valid_answer_and_standard_json(valid_answer, capsys):
    source = json.dumps(valid_answer, ensure_ascii=False)
    result = parse_analysis_answer(source)
    assert result == valid_answer
    assert json.loads(json.dumps(result, allow_nan=False)) == valid_answer
    output = capsys.readouterr()
    assert output.out == output.err == ""


def test_empty_explanation_and_missing_information_are_valid():
    answer = {"summary": "概念解释", "explanation": "", "missing_information": []}
    assert parse_analysis_answer(json.dumps(answer)) == answer


def test_no_silent_string_cleaning(valid_answer):
    valid_answer["summary"] = "  结论  "
    valid_answer["explanation"] = '包含"引号"和\n换行'
    assert parse_analysis_answer(json.dumps(valid_answer)) == valid_answer


@pytest.mark.parametrize(
    "source",
    [
        "",
        "not JSON",
        '{"summary":',
        '```json\n{"summary":"x","explanation":"","missing_information":[]}\n```',
        '{"summary":"x","explanation":"","missing_information":[]} additional text',
    ],
)
def test_invalid_json_preserves_decode_error(source):
    with pytest.raises(json.JSONDecodeError) as caught:
        parse_analysis_answer(source)
    assert caught.value.doc == source


@pytest.mark.parametrize("value", [[], "text", 1, True, None])
def test_root_must_be_object(value):
    with pytest.raises(ValueError, match="JSON 对象"):
        parse_analysis_answer(json.dumps(value))


@pytest.mark.parametrize("field", ["summary", "explanation", "missing_information"])
def test_required_fields_are_not_filled(valid_answer, field):
    del valid_answer[field]
    with pytest.raises(ValueError, match=field):
        parse_analysis_answer(json.dumps(valid_answer))


def test_extra_fields_are_not_discarded(valid_answer):
    valid_answer["unexpected"] = "extra"
    with pytest.raises(ValueError, match="unexpected"):
        parse_analysis_answer(json.dumps(valid_answer))


@pytest.mark.parametrize("value", [None, 1, False, [], {}])
@pytest.mark.parametrize("field", ["summary", "explanation"])
def test_text_fields_reject_other_types(valid_answer, field, value):
    valid_answer[field] = value
    with pytest.raises(ValueError, match=field):
        parse_analysis_answer(json.dumps(valid_answer))


@pytest.mark.parametrize("summary", ["", " ", "\n\t", "　"])
def test_summary_cannot_be_blank(valid_answer, summary):
    valid_answer["summary"] = summary
    with pytest.raises(ValueError, match="summary"):
        parse_analysis_answer(json.dumps(valid_answer))


@pytest.mark.parametrize("value", [None, "data", 1, True, {}])
def test_missing_information_must_be_list(valid_answer, value):
    valid_answer["missing_information"] = value
    with pytest.raises(ValueError, match="missing_information"):
        parse_analysis_answer(json.dumps(valid_answer))


@pytest.mark.parametrize("item", [None, 1, False, [], {}])
def test_missing_information_items_must_be_strings(valid_answer, item):
    valid_answer["missing_information"] = ["valid item", item]
    with pytest.raises(ValueError, match="每个元素"):
        parse_analysis_answer(json.dumps(valid_answer))
