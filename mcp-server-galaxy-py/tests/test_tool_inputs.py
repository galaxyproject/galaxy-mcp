import bioblend

from galaxy_mcp.tool_inputs import (
    build_input_template,
    is_input_related_error,
    summarize_tool_inputs,
)


def _conn_err(status, body="boom"):
    return bioblend.ConnectionError(
        f"Unexpected HTTP status code: {status}", body=body, status_code=status
    )


def test_400_is_input_related():
    assert is_input_related_error(_conn_err(400, "Required parameter(s) kwd not provided.")) is True


def test_typeerror_is_input_related():
    assert is_input_related_error(TypeError("bad inputs")) is True


def test_auth_and_notfound_are_not_input_related():
    assert is_input_related_error(_conn_err(401)) is False
    assert is_input_related_error(_conn_err(403)) is False
    assert is_input_related_error(_conn_err(404)) is False
    assert is_input_related_error(_conn_err(500)) is False


def test_plain_exception_is_not_input_related():
    assert is_input_related_error(RuntimeError("network down")) is False


def test_valueerror_is_not_input_related():
    assert is_input_related_error(ValueError("bad inputs")) is False


# Shape mirrors Galaxy's /api/tools/{id}?io_details=true output.
BUILD_LIST_SCHEMA = {
    "id": "__BUILD_LIST__",
    "inputs": [
        {
            "name": "datasets",
            "type": "repeat",
            "inputs": [
                {"name": "input", "type": "data", "optional": False},
                {
                    "name": "id_cond",
                    "type": "conditional",
                    "test_param": {
                        "name": "id_select",
                        "type": "select",
                        "options": [["use index", "idx", True], ["manual", "manual", False]],
                    },
                    "cases": [
                        {"value": "idx", "inputs": []},
                        {"value": "manual", "inputs": [{"name": "identifier", "type": "text"}]},
                    ],
                },
            ],
        }
    ],
}


def test_summarize_walks_repeat_conditional():
    summ = summarize_tool_inputs(BUILD_LIST_SCHEMA)
    assert summ[0]["name"] == "datasets"
    assert summ[0]["type"] == "repeat"
    assert summ[0]["repeat_key_hint"] == "datasets_0|<param>"
    child_names = [c["name"] for c in summ[0]["children"]]
    assert child_names == ["input", "id_cond"]
    cond = summ[0]["children"][1]
    assert cond["type"] == "conditional"
    assert cond["selector"]["name"] == "id_select"
    assert {c["when"] for c in cond["cases"]} == {"idx", "manual"}


def test_summarize_handles_missing_inputs_key():
    assert summarize_tool_inputs({}) == []
    assert summarize_tool_inputs({"inputs": []}) == []


CAT1_SCHEMA = {
    "id": "cat1",
    "inputs": [
        {"name": "input1", "type": "data", "optional": False},
        {"name": "queries", "type": "repeat", "inputs": [{"name": "input2", "type": "data"}]},
    ],
}


def test_template_flattens_data_and_repeat():
    t = build_input_template(CAT1_SCHEMA)
    assert t["input1"] == {"src": "hda", "id": "<dataset_id>"}
    assert t["queries_0|input2"] == {"src": "hda", "id": "<dataset_id>"}


def test_template_conditional_uses_first_case():
    t = build_input_template(BUILD_LIST_SCHEMA)
    assert t["datasets_0|input"] == {"src": "hda", "id": "<dataset_id>"}
    # first case is "idx" -> selector set, no extra params
    assert t["datasets_0|id_cond|id_select"] == "idx"
    assert "datasets_0|id_cond|identifier" not in t


def test_template_non_dict_returns_empty():
    assert build_input_template(None) == {}
    assert build_input_template("not a dict") == {}


def test_template_section_flattens():
    schema = {
        "inputs": [{"name": "adv", "type": "section", "inputs": [{"name": "p", "type": "integer"}]}]
    }
    assert build_input_template(schema)["adv|p"] == 0
