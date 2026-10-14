"""Local Pydantic validation tests; no model or API calls."""

import json
import runpy
import socket
from pathlib import Path

import pytest
from pydantic import ValidationError

from data_copilot.schemas import AnalysisAnswer


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbid_network(*args, **kwargs):
        raise AssertionError("Pydantic experiments must not use the network")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)


@pytest.fixture
def valid_data():
    return {
        "summary": "  概念解释  ",
        "explanation": '说明包含"引号"和\n换行',
        "missing_information": ["销售额数据"],
    }


@pytest.fixture(params=["dict", "json"])
def validate(request):
    """Exercise both public entry points without changing their semantics."""
    if request.param == "dict":
        return AnalysisAnswer.model_validate
    return lambda data: AnalysisAnswer.model_validate_json(
        json.dumps(data, ensure_ascii=False)
    )


def test_valid_model_round_trip_preserves_text(validate, valid_data, capsys):
    model = validate(valid_data)
    assert isinstance(model, AnalysisAnswer)
    assert model.summary == valid_data["summary"]
    assert model.model_dump() == valid_data
    assert json.loads(model.model_dump_json()) == valid_data
    assert json.loads(json.dumps(model.model_dump(), allow_nan=False)) == valid_data
    output = capsys.readouterr()
    assert output.out == output.err == ""


def test_empty_explanation_and_list_are_valid(validate, valid_data):
    valid_data.update(explanation="", missing_information=[])
    assert validate(valid_data).model_dump() == valid_data


@pytest.mark.parametrize("field", ["summary", "explanation", "missing_information"])
def test_missing_fields_are_required(validate, valid_data, field):
    del valid_data[field]
    with pytest.raises(ValidationError) as caught:
        validate(valid_data)
    assert any(
        error["loc"] == (field,) and error["type"] == "missing"
        for error in caught.value.errors()
    )


def test_extra_field_is_forbidden(validate, valid_data):
    valid_data["confidence"] = 0.9
    with pytest.raises(ValidationError) as caught:
        validate(valid_data)
    assert any(
        error["loc"] == ("confidence",) and error["type"] == "extra_forbidden"
        for error in caught.value.errors()
    )


@pytest.mark.parametrize("field", ["summary", "explanation"])
@pytest.mark.parametrize("value", [123, True, None])
def test_text_fields_do_not_coerce_types(validate, valid_data, field, value):
    valid_data[field] = value
    with pytest.raises(ValidationError) as caught:
        validate(valid_data)
    assert any(
        error["loc"] == (field,) and error["type"] == "string_type"
        for error in caught.value.errors()
    )


@pytest.mark.parametrize("summary", ["", " \n\t", "　"])
def test_blank_summary_is_rejected(validate, valid_data, summary):
    valid_data["summary"] = summary
    with pytest.raises(ValidationError) as caught:
        validate(valid_data)
    assert any(
        error["loc"] == ("summary",)
        and error["type"] == "value_error"
        and "不能为空" in error["msg"]
        for error in caught.value.errors()
    )


@pytest.mark.parametrize("value", [None, "data"])
def test_missing_information_requires_list(validate, valid_data, value):
    valid_data["missing_information"] = value
    with pytest.raises(ValidationError) as caught:
        validate(valid_data)
    assert any(
        error["loc"] == ("missing_information",) and error["type"] == "list_type"
        for error in caught.value.errors()
    )


@pytest.mark.parametrize("item", [42, None])
def test_list_elements_must_be_strings(validate, valid_data, item):
    valid_data["missing_information"] = ["valid", item]
    with pytest.raises(ValidationError) as caught:
        validate(valid_data)
    assert any(
        error["loc"] == ("missing_information", 1) and error["type"] == "string_type"
        for error in caught.value.errors()
    )


def test_python_tuple_is_not_silently_converted_to_list(valid_data):
    valid_data["missing_information"] = ("data",)
    with pytest.raises(ValidationError) as caught:
        AnalysisAnswer.model_validate(valid_data)
    assert caught.value.errors()[0]["type"] == "list_type"


@pytest.mark.parametrize("source", ["", '{"summary":', "```json\n{}\n```"])
def test_invalid_json_raises_pydantic_validation_error(source):
    with pytest.raises(ValidationError) as caught:
        AnalysisAnswer.model_validate_json(source)
    assert caught.value.errors()[0]["type"] == "json_invalid"


def test_generated_schema_has_required_fields_and_forbids_extras():
    schema = AnalysisAnswer.model_json_schema()
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"summary", "explanation", "missing_information"}
    assert schema["additionalProperties"] is False
    properties = schema["properties"]
    assert properties["summary"]["type"] == "string"
    assert properties["explanation"]["type"] == "string"
    assert properties["missing_information"]["type"] == "array"
    assert properties["missing_information"]["items"]["type"] == "string"
    assert all(field.is_required() for field in AnalysisAnswer.model_fields.values())
    json.dumps(schema, allow_nan=False)


def test_local_experiment_runs_and_rejects_all_invalid_examples(capsys):
    path = Path(__file__).resolve().parents[1] / "examples/pydantic_validation.py"
    namespace = runpy.run_path(str(path), run_name="__main__")
    output = capsys.readouterr()
    assert output.err == ""
    for method in [
        "model_validate",
        "model_validate_json",
        "model_dump",
        "model_dump_json",
        "model_json_schema",
    ]:
        assert f"[{method}]" in output.out
    for case_name in namespace["INVALID_ANSWERS"]:
        assert f"[{case_name}] 拒绝：" in output.out
