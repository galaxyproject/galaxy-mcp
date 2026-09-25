import bioblend
import pytest

from galaxy_mcp.ops.tool_inputs import (
    ToolInputsUncheckableError,
    _lookup_param,
    build_input_template,
    check_tool_inputs,
    classify_src,
    describe_reference,
    format_input_mismatch_error,
    index_tool_params,
    is_input_related_error,
    schema_describes_tool,
    schema_has_inputs,
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


def test_summarize_select_truncates_long_option_lists():
    schema = {
        "inputs": [
            {
                "name": "genome",
                "type": "select",
                "options": [[f"label{i}", f"val{i}", False] for i in range(40)],
            }
        ]
    }
    param = summarize_tool_inputs(schema)[0]
    assert param["choices"][:3] == ["val0", "val1", "val2"]
    assert len(param["choices"]) == 25  # capped at _MAX_OPTIONS, values only
    assert param["choices_truncated"] is True


def test_summarize_select_keeps_short_option_lists_intact():
    schema = {
        "inputs": [
            {"name": "fmt", "type": "select", "options": [["A", "a", False], ["B", "b", True]]}
        ]
    }
    param = summarize_tool_inputs(schema)[0]
    assert param["choices"] == ["a", "b"]
    assert "choices_truncated" not in param


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


def test_message_includes_schema_and_disclaimer_and_original():
    msg = format_input_mismatch_error(
        original_error="Run tool failed: Unexpected HTTP status code: 400: kwd not provided",
        tool_id="cat1",
        schema_summary=[{"name": "input1", "type": "data"}],
        example={"input1": {"src": "hda", "id": "1"}},
    )
    assert "Run tool failed" in msg  # original preserved verbatim
    assert "not a sign" in msg.lower()  # disclaimer
    assert "kwd" in msg.lower()  # names the misleading wording
    assert "input1" in msg  # schema embedded
    assert "NOT runnable" in msg  # example caveat
    assert "cat1" in msg


def test_message_degrades_without_schema_or_example():
    msg = format_input_mismatch_error(
        original_error="Run tool failed: boom",
        tool_id="cat1",
        schema_summary=None,
        example=None,
    )
    assert "get_tool_details" in msg
    assert "Run tool failed: boom" in msg


# ---------------------------------------------------------------------------
# Input preflight: classifying supplied references and indexing the schema
# ---------------------------------------------------------------------------


def test_classify_src_buckets_every_kind():
    assert classify_src("hda") == "dataset"
    # A library dataset is still one dataset, not a collection.
    assert classify_src("ldda") == "dataset"
    assert classify_src("ld") == "dataset"
    assert classify_src("hdca") == "collection"
    assert classify_src("dce") == "element"
    assert classify_src("something_new") == "unknown"
    assert classify_src(None) == "unknown"
    assert classify_src(7) == "unknown"


def test_describe_reference_uses_the_workflow_validator_vocabulary():
    assert describe_reference("hda") == "a single dataset (hda)"
    assert describe_reference("hdca") == "a dataset collection (hdca)"
    assert describe_reference("dce") == "a collection element (dce)"
    assert describe_reference("mystery") == "src 'mystery'"


NESTED_TOOL = {
    "id": "nested",
    "inputs": [
        {"name": "plain", "type": "data"},
        {
            "name": "sect",
            "type": "section",
            "inputs": [{"name": "inner", "type": "data"}],
        },
        {
            "name": "rep",
            "type": "repeat",
            "inputs": [{"name": "item", "type": "data"}],
        },
        {
            "name": "cond",
            "type": "conditional",
            "test_param": {"name": "selector", "type": "select", "options": [["A", "a", False]]},
            "cases": [
                {"value": "a", "inputs": [{"name": "branch", "type": "data"}]},
                {"value": "b", "inputs": [{"name": "branch", "type": "data_collection"}]},
            ],
        },
    ],
}


def test_index_tool_params_covers_every_nesting_form():
    index = index_tool_params(NESTED_TOOL)

    assert set(index) == {"plain", "sect|inner", "rep_#|item", "cond|selector", "cond|branch"}
    assert index["plain"][0]["type"] == "data"
    assert index["sect|inner"][0]["type"] == "data"
    assert index["rep_#|item"][0]["type"] == "data"


def test_index_keeps_every_conditional_branch_for_the_same_key():
    """The same key means different things per case; the checker needs both."""
    index = index_tool_params(NESTED_TOOL)

    assert [p["type"] for p in index["cond|branch"]] == ["data", "data_collection"]


def test_lookup_resolves_any_repeat_instance_index():
    index = index_tool_params(NESTED_TOOL)

    assert _lookup_param("rep_0|item", index) == index["rep_#|item"]
    assert _lookup_param("rep_11|item", index) == index["rep_#|item"]
    assert _lookup_param("plain", index) == index["plain"]
    assert _lookup_param("nope", index) == []


def test_index_tolerates_junk_schemas():
    assert index_tool_params({}) == {}
    assert index_tool_params({"inputs": None}) == {}
    assert index_tool_params({"inputs": ["not a dict", {"no_name": 1}]}) == {}
    assert index_tool_params("not a dict") == {}


def test_schema_describes_tool_matches_exact_and_resolved_ids():
    assert schema_describes_tool("cat1", {"id": "cat1"}) is True
    # An unversioned request resolves to the installed version's full id.
    assert (
        schema_describes_tool(
            "ts/repos/iuc/fastqc/fastqc", {"id": "ts/repos/iuc/fastqc/fastqc/0.74"}
        )
        is True
    )


def test_schema_describes_tool_rejects_a_different_version():
    """Asking for 0.73 and being handed 0.74 must not be treated as a match."""
    assert (
        schema_describes_tool(
            "ts/repos/iuc/fastqc/fastqc/0.73", {"id": "ts/repos/iuc/fastqc/fastqc/0.74"}
        )
        is False
    )
    assert schema_describes_tool("cat1", {}) is False
    assert schema_describes_tool("cat1", {"id": ""}) is False
    assert schema_describes_tool("cat1", None) is False


# ---------------------------------------------------------------------------
# Input preflight: what Galaxy actually refuses, and what it does not
# ---------------------------------------------------------------------------
#
# Rules verified against galaxyproject/galaxy release_24.2 through dev:
#  - a bare hdca on a multiple=false data param fails on every version
#    (clean 400 since 26.0, an execution error before that)
#  - a bare hdca on multiple=true is the REDUCE form and is valid
#  - {"batch": true, "values": [...]} is map-over and is valid
#  - hda/ldda on a data_collection param is refused
#  - collection_type and extension mismatches are NOT checked at submission

SINGLE_DATA = {
    "id": "cat1",
    "version": "1.0.0",
    "inputs": [{"name": "input1", "type": "data", "multiple": False, "extensions": ["txt"]}],
}
MULTI_DATA = {
    "id": "multi",
    "inputs": [{"name": "queries", "type": "data", "multiple": True, "extensions": ["txt"]}],
}
COLLECTION_TOOL = {
    "id": "coll",
    "inputs": [
        {"name": "coll_in", "type": "data_collection", "collection_types": ["list"]},
    ],
}


def rejects(tool_info, inputs):
    return check_tool_inputs(tool_info, inputs)["rejects"]


def test_hda_into_a_single_data_param_is_accepted():
    assert rejects(SINGLE_DATA, {"input1": {"src": "hda", "id": "d1"}}) == []


def test_library_dataset_into_a_single_data_param_is_accepted():
    """ldda is a single dataset; the workflow validator calls it a collection."""
    assert rejects(SINGLE_DATA, {"input1": {"src": "ldda", "id": "l1"}}) == []


def test_collection_into_a_single_data_param_is_refused():
    found = rejects(SINGLE_DATA, {"input1": {"src": "hdca", "id": "c1"}})

    assert len(found) == 1
    assert found[0]["param"] == "input1"
    reason = found[0]["reason"]
    assert "single dataset" in reason
    assert "collection" in reason
    assert "batch" in reason  # the remedy, not just the diagnosis


def test_collection_mapped_over_a_single_data_param_is_accepted():
    """The batch wrapper is how map-over is spelled, and it is valid."""
    value = {"batch": True, "values": [{"src": "hdca", "id": "c1"}]}

    assert rejects(SINGLE_DATA, {"input1": value}) == []


def test_subcollection_map_over_is_accepted():
    value = {"batch": True, "values": [{"src": "hdca", "id": "c1", "map_over_type": "list"}]}

    assert rejects(SINGLE_DATA, {"input1": value}) == []


def test_collection_element_into_a_single_data_param_is_accepted():
    """A dce feeding a data param is the legitimate nested map-over shape."""
    assert rejects(SINGLE_DATA, {"input1": {"src": "dce", "id": "e1"}}) == []


def test_collection_into_a_multiple_data_param_is_accepted():
    """Bare hdca on multiple=true is Galaxy's reduce form."""
    assert rejects(MULTI_DATA, {"queries": {"src": "hdca", "id": "c1"}}) == []


def test_list_of_datasets_into_a_multiple_data_param_is_accepted():
    value = [{"src": "hda", "id": "d1"}, {"src": "hda", "id": "d2"}]

    assert rejects(MULTI_DATA, {"queries": value}) == []


def test_list_of_collections_into_a_multiple_data_param_is_accepted():
    value = [{"src": "hdca", "id": "c1"}, {"src": "hdca", "id": "c2"}]

    assert rejects(MULTI_DATA, {"queries": value}) == []


def test_list_mixing_a_collection_with_a_dataset_is_refused():
    value = [{"src": "hda", "id": "d1"}, {"src": "hdca", "id": "c1"}]

    found = rejects(MULTI_DATA, {"queries": value})

    assert len(found) == 1
    assert "only" in found[0]["reason"] or "every value must be a collection" in found[0]["reason"]


def test_dataset_into_a_collection_param_is_refused():
    found = rejects(COLLECTION_TOOL, {"coll_in": {"src": "hda", "id": "d1"}})

    assert len(found) == 1
    assert found[0]["param"] == "coll_in"
    assert "dataset collection" in found[0]["reason"]


def test_collection_into_a_collection_param_is_accepted():
    assert rejects(COLLECTION_TOOL, {"coll_in": {"src": "hdca", "id": "c1"}}) == []


def test_mismatched_collection_type_is_not_refused():
    """Galaxy does not enforce collection_type at submission, so neither do we."""
    assert rejects(COLLECTION_TOOL, {"coll_in": {"src": "hdca", "id": "c1"}}) == []


def test_mismatched_extension_is_not_refused():
    """from_json computes a datatype match but never rejects on it."""
    assert rejects(SINGLE_DATA, {"input1": {"src": "hda", "id": "d1", "ext": "bam"}}) == []


def test_non_reference_values_are_left_alone():
    """Bare ids, legacy strings and nulls are all things Galaxy accepts."""
    for value in ("123", 456, "__collection_reduce__|abc", "hdca:12", None, ""):
        assert rejects(SINGLE_DATA, {"input1": value}) == []


def test_unknown_src_is_left_alone():
    assert rejects(SINGLE_DATA, {"input1": {"src": "CollectionAdapter", "adapting": {}}}) == []


def test_absent_multiple_flag_means_one_dataset():
    """galaxy-tool-util's DataParameterModel defaults multiple to False."""
    schema = {"id": "x", "inputs": [{"name": "input1", "type": "data"}]}

    assert rejects(schema, {"input1": {"src": "hdca", "id": "c1"}})
    assert rejects(schema, {"input1": {"src": "hda", "id": "d1"}}) == []


def test_nested_conditional_param_is_checked():
    schema = {
        "id": "nested",
        "inputs": [
            {
                "name": "outer",
                "type": "conditional",
                "test_param": {"name": "sel", "type": "select"},
                "cases": [
                    {
                        "value": "single",
                        "inputs": [{"name": "inp", "type": "data", "multiple": False}],
                    }
                ],
            }
        ],
    }

    assert rejects(schema, {"outer|inp": {"src": "hdca", "id": "c1"}})[0]["param"] == "outer|inp"
    assert rejects(schema, {"outer|inp": {"src": "hda", "id": "d1"}}) == []


def test_a_key_meaning_different_things_per_branch_is_left_alone():
    """One branch would refuse the collection, the other wants one; stay quiet."""
    schema = {
        "id": "ambiguous",
        "inputs": [
            {
                "name": "cond",
                "type": "conditional",
                "test_param": {"name": "sel", "type": "select"},
                "cases": [
                    {"value": "a", "inputs": [{"name": "x", "type": "data", "multiple": False}]},
                    {"value": "b", "inputs": [{"name": "x", "type": "data_collection"}]},
                ],
            }
        ],
    }

    assert rejects(schema, {"cond|x": {"src": "hdca", "id": "c1"}}) == []


def test_repeat_instance_is_checked_at_any_index():
    schema = {
        "id": "rep",
        "inputs": [
            {
                "name": "rep",
                "type": "repeat",
                "inputs": [{"name": "item", "type": "data", "multiple": False}],
            }
        ],
    }

    assert rejects(schema, {"rep_0|item": {"src": "hdca", "id": "c1"}})[0]["param"] == "rep_0|item"
    assert rejects(schema, {"rep_9|item": {"src": "hdca", "id": "c1"}})[0]["param"] == "rep_9|item"


def test_unknown_parameter_names_warn_but_never_block():
    verdict = check_tool_inputs(SINGLE_DATA, {"not_a_param": {"src": "hda", "id": "d1"}})

    assert verdict["rejects"] == []
    assert verdict["warnings"][0]["param"] == "not_a_param"


def test_checker_never_raises_on_junk():
    for tool_info in ({}, {"inputs": None}, "nope", None):
        assert check_tool_inputs(tool_info, {"a": 1})["rejects"] == []
    assert check_tool_inputs(SINGLE_DATA, "not a dict") == {"rejects": [], "warnings": []}
    assert check_tool_inputs(SINGLE_DATA, {1: "non-string key"})["rejects"] == []


@pytest.mark.parametrize(
    "inputs",
    [
        {"q": [{"src": ["hdca"], "id": "x"}]},  # an unhashable src
        {"q": [{"src": {"a": 1}, "id": "x"}]},
        {"q": {"src": 7, "id": "x"}},
    ],
)
def test_checker_never_raises_on_a_malformed_src(inputs):
    """A caller sending nonsense must still reach Galaxy's own error, not a traceback."""
    assert check_tool_inputs(MULTI_DATA, inputs)["rejects"] == []


@pytest.mark.parametrize(
    "broken",
    [
        {"id": "t", "inputs": [{"name": "c", "type": "conditional", "test_param": "notadict"}]},
        {"id": "t", "inputs": [{"name": "c", "type": "conditional", "cases": 5}]},
        {"id": "t", "inputs": [{"name": "r", "type": "repeat", "inputs": "nope"}]},
    ],
)
def test_checker_never_raises_on_a_malformed_schema(broken):
    assert check_tool_inputs(broken, {"x": {"src": "hdca", "id": "c"}})["rejects"] == []


def test_schema_has_inputs_separates_no_inputs_from_no_input_list():
    assert schema_has_inputs({"id": "t", "inputs": []}) is True
    assert schema_has_inputs({"id": "t", "inputs": [{"name": "a", "type": "data"}]}) is True
    # Fetched without io_details: not checkable, and not the same as "no inputs".
    assert schema_has_inputs({"id": "t"}) is False
    assert schema_has_inputs({"id": "t", "inputs": None}) is False
    assert schema_has_inputs("nope") is False


# ---------------------------------------------------------------------------
# The false-reject guard: every shape Galaxy is known to accept, in one table
# ---------------------------------------------------------------------------

NESTED_REPEAT = {
    "id": "t",
    "inputs": [
        {
            "name": "outer",
            "type": "repeat",
            "inputs": [
                {
                    "name": "inner",
                    "type": "repeat",
                    "inputs": [{"name": "d", "type": "data", "multiple": False}],
                }
            ],
        }
    ],
}
SECTION_IN_CONDITIONAL = {
    "id": "t",
    "inputs": [
        {
            "name": "c",
            "type": "conditional",
            "test_param": {"name": "s", "type": "select"},
            "cases": [
                {
                    "value": "a",
                    "inputs": [
                        {
                            "name": "sec",
                            "type": "section",
                            "inputs": [{"name": "d", "type": "data", "multiple": False}],
                        }
                    ],
                }
            ],
        }
    ],
}
PARAM_NAMED_LIKE_A_REPEAT = {
    "id": "t",
    "inputs": [{"name": "q_1", "type": "data", "multiple": False}],
}

REPEAT_WITH_INDEXED_CHILD = {
    "id": "t",
    "inputs": [
        {
            "name": "queries",
            "type": "repeat",
            "inputs": [
                {"name": "input_1", "type": "data", "multiple": False},
                {"name": "label", "type": "text"},
            ],
        }
    ],
}


@pytest.mark.parametrize(
    ("label", "schema", "inputs"),
    [
        ("single dataset", SINGLE_DATA, {"input1": {"src": "hda", "id": "x"}}),
        ("library dataset", SINGLE_DATA, {"input1": {"src": "ldda", "id": "x"}}),
        ("collection element", SINGLE_DATA, {"input1": {"src": "dce", "id": "x"}}),
        (
            "map-over",
            SINGLE_DATA,
            {"input1": {"batch": True, "values": [{"src": "hdca", "id": "x"}]}},
        ),
        (
            "cartesian map-over",
            SINGLE_DATA,
            {"input1": {"batch": True, "linked": False, "values": [{"src": "hdca", "id": "x"}]}},
        ),
        (
            "values wrapper without batch",
            SINGLE_DATA,
            {"input1": {"values": [{"src": "hda", "id": "x"}]}},
        ),
        (
            "multi-run over datasets",
            SINGLE_DATA,
            {
                "input1": {
                    "batch": True,
                    "values": [{"src": "hda", "id": "a"}, {"src": "hda", "id": "b"}],
                }
            },
        ),
        (
            "collection adapter",
            SINGLE_DATA,
            {"input1": {"src": "CollectionAdapter", "adapting": {}}},
        ),
        ("bare numeric id", SINGLE_DATA, {"input1": 42}),
        (
            "extra keys in the src dict",
            SINGLE_DATA,
            {"input1": {"src": "hda", "id": "x", "hid": 3}},
        ),
        ("nested 21.01-style dict", SINGLE_DATA, {"input1": {"foo": "bar"}}),
        ("comma-joined ids", MULTI_DATA, {"queries": "12,34"}),
        ("legacy reduce spelling", MULTI_DATA, {"queries": "__collection_reduce__|abc"}),
        ("reduce a collection", MULTI_DATA, {"queries": {"src": "hdca", "id": "x"}}),
        (
            "reduce several collections",
            MULTI_DATA,
            {"queries": [{"src": "hdca", "id": "a"}, {"src": "hdca", "id": "b"}]},
        ),
        (
            "a bare id alongside a collection",
            MULTI_DATA,
            {"queries": ["12", {"src": "hdca", "id": "a"}]},
        ),
        (
            "collection into a collection param",
            COLLECTION_TOOL,
            {"coll_in": {"src": "hdca", "id": "x"}},
        ),
        (
            "element into a collection param",
            COLLECTION_TOOL,
            {"coll_in": {"src": "dce", "id": "x"}},
        ),
        (
            "collection of the wrong shape",
            COLLECTION_TOOL,
            {"coll_in": {"src": "hdca", "id": "x", "collection_type": "paired"}},
        ),
        ("nested repeats", NESTED_REPEAT, {"outer_2|inner_5|d": {"src": "hda", "id": "x"}}),
        (
            "section inside a conditional",
            SECTION_IN_CONDITIONAL,
            {"c|sec|d": {"src": "hda", "id": "x"}},
        ),
        (
            "a real param whose name ends in an index",
            PARAM_NAMED_LIKE_A_REPEAT,
            {"q_1": {"src": "hda", "id": "x"}},
        ),
        (
            "a repeat child whose name ends in an index",
            REPEAT_WITH_INDEXED_CHILD,
            {"queries_0|input_1": {"src": "hda", "id": "x"}},
        ),
        ("a key Galaxy reads itself", SINGLE_DATA, {"dbkey": "hg38"}),
        (
            "the job-resource conditional Galaxy injects",
            SINGLE_DATA,
            {"__job_resource|__job_resource__select": "no"},
        ),
    ],
)
def test_shapes_galaxy_accepts_are_never_refused(label, schema, inputs):
    """A false reject blocks a run that works; it is worse than no check at all."""
    assert rejects(schema, inputs) == [], label


@pytest.mark.parametrize(
    ("label", "schema", "inputs"),
    [
        ("collection into single data", SINGLE_DATA, {"input1": {"src": "hdca", "id": "x"}}),
        ("inside nested repeats", NESTED_REPEAT, {"outer_2|inner_5|d": {"src": "hdca", "id": "x"}}),
        (
            "inside a section in a conditional",
            SECTION_IN_CONDITIONAL,
            {"c|sec|d": {"src": "hdca", "id": "x"}},
        ),
        (
            "collection mixed with a dataset",
            MULTI_DATA,
            {"queries": [{"src": "hda", "id": "a"}, {"src": "hdca", "id": "b"}]},
        ),
        (
            "dataset into a collection param",
            COLLECTION_TOOL,
            {"coll_in": {"src": "hda", "id": "x"}},
        ),
        (
            "library dataset into a collection param",
            COLLECTION_TOOL,
            {"coll_in": {"src": "ldda", "id": "x"}},
        ),
        (
            "library dataset id into a collection param",
            COLLECTION_TOOL,
            {"coll_in": {"src": "ld", "id": "x"}},
        ),
    ],
)
def test_shapes_galaxy_refuses_are_caught(label, schema, inputs):
    assert rejects(schema, inputs), label


# ---------------------------------------------------------------------------
# Keys the caller supplies that the schema does not describe
# ---------------------------------------------------------------------------
#
# Two ways the old code told an agent to delete a working input: a repeat child
# whose own name ends in an index never resolved, and the keys Galaxy reads for
# itself were listed as suspected typos.


def test_repeat_child_named_like_an_index_still_resolves():
    """queries_0|input_1 is a real parameter; calling it unrecognised is the bug."""
    verdict = check_tool_inputs(
        REPEAT_WITH_INDEXED_CHILD, {"queries_0|input_1": {"src": "hda", "id": "d1"}}
    )

    assert verdict == {"rejects": [], "warnings": []}


def test_repeat_child_named_like_an_index_is_checked_like_any_other():
    found = rejects(REPEAT_WITH_INDEXED_CHILD, {"queries_3|input_1": {"src": "hdca", "id": "c1"}})

    assert [r["param"] for r in found] == ["queries_3|input_1"]


def test_section_child_named_like_an_index_resolves():
    schema = {
        "id": "t",
        "inputs": [
            {
                "name": "adv",
                "type": "section",
                "inputs": [{"name": "bed_2", "type": "data", "multiple": False}],
            }
        ],
    }

    assert check_tool_inputs(schema, {"adv|bed_2": {"src": "hda", "id": "d1"}})["warnings"] == []


@pytest.mark.parametrize(
    "key",
    [
        "dbkey",
        "chromInfo",
        "use_cached_job",
        "send_email_notification",
        "rerun_remap_job_id",
        "__job_resource",
        "__job_resource|__job_resource__select",
        "__job_resource|cores",
    ],
)
def test_keys_galaxy_handles_itself_are_not_reported_as_unrecognised(key):
    """Galaxy handles these outside the tool schema; naming them costs the agent a run."""
    verdict = check_tool_inputs(SINGLE_DATA, {"input1": {"src": "hda", "id": "d1"}, key: "value"})

    assert verdict == {"rejects": [], "warnings": []}


def test_a_genuine_unknown_key_is_still_reported():
    verdict = check_tool_inputs(SINGLE_DATA, {"dbky": "hg38"})

    assert [w["param"] for w in verdict["warnings"]] == ["dbky"]


def test_a_tool_that_declares_dbkey_itself_is_checked_normally():
    """Some tools declare a dbkey param; the skip list must not shadow a real one."""
    schema = {
        "id": "t",
        "inputs": [{"name": "dbkey", "type": "data", "multiple": False}],
    }

    assert rejects(schema, {"dbkey": {"src": "hdca", "id": "c1"}})


def test_the_unrecognised_wording_does_not_call_them_typos():
    msg = format_input_mismatch_error(
        original_error="400",
        tool_id="cat1",
        schema_summary=None,
        example=None,
        unmodelled=["wat"],
    )

    assert "not in this tool's parameter list" in msg
    assert "outside the tool schema" in msg
    assert "typo" not in msg


# ---------------------------------------------------------------------------
# A dataset src on a collection parameter: three srcs, three different fates
# ---------------------------------------------------------------------------


def _collection_reason(src):
    found = rejects(COLLECTION_TOOL, {"coll_in": {"src": src, "id": "x"}})
    assert len(found) == 1, src
    return found[0]["reason"]


def test_hda_on_a_collection_param_is_named_as_the_400_it_is():
    reason = _collection_reason("hda")

    assert "400" in reason
    assert "Expected to find collection" in reason


def test_ldda_on_a_collection_param_says_the_job_starts_without_it():
    """_patch_library_inputs copies it into the history first; Galaxy answers 200."""
    reason = _collection_reason("ldda")

    assert "empty" in reason
    assert "without the collection" in reason
    # The accepted-but-broken outcome is the headline; it must not read as a refusal.
    assert reason.index("without the collection") < reason.index("400")


def test_ldda_on_a_collection_param_still_mentions_the_unreadable_case():
    """_patch_library_dataset only rewrites a library dataset the caller can read."""
    reason = _collection_reason("ldda")

    assert "readable" in reason
    assert "cannot read 400s" in reason


def test_ld_on_a_collection_param_says_the_tool_api_cannot_resolve_it():
    """src_id_to_item has no 'ld'; only the workflow API resolves a library dataset id."""
    reason = _collection_reason("ld")

    assert "Unknown input source ld" in reason
    assert "workflow API" in reason


# ---------------------------------------------------------------------------
# Repeat instances Galaxy would never read
# ---------------------------------------------------------------------------
#
# _populate_state_legacy fills a repeat from index 0 and breaks at the first index
# no supplied key mentions, so a sparse index is dropped silently. We still check
# the value -- staying silent while the caller's input disappears is worse -- but
# the refusal has to tell them the numbering is the thing to fix.

REPEAT_TOOL = {
    "id": "t",
    "inputs": [
        {
            "name": "rep",
            "type": "repeat",
            "inputs": [
                {"name": "item", "type": "data", "multiple": False},
                {"name": "label", "type": "text"},
            ],
        }
    ],
}
_HINT = "number this tool's repeat instances from 0"


def test_a_bad_value_at_an_unreachable_index_is_also_told_to_renumber():
    """The advice rides on the type reject; the index alone is not a reason to refuse."""
    found = rejects(REPEAT_TOOL, {"rep_5|item": {"src": "hdca", "id": "c1"}})

    assert _HINT in found[0]["reason"]


def test_a_good_value_at_an_unreachable_index_is_left_alone():
    """Galaxy drops the instance and runs the job, so there is nothing to refuse."""
    verdict = check_tool_inputs(REPEAT_TOOL, {"rep_5|item": {"src": "hda", "id": "d1"}})

    assert verdict == {"rejects": [], "warnings": []}


def test_an_unreachable_index_is_never_a_reject_on_its_own():
    """Whatever else it holds, a gap does not turn a usable value into a refusal."""
    for value in ({"src": "hda", "id": "d1"}, {"src": "dce", "id": "e1"}, "plain text"):
        assert rejects(REPEAT_TOOL, {"rep_9|item": value}) == [], value
    assert rejects(REPEAT_TOOL, {"rep_9|label": "text"}) == []


def test_index_zero_gets_no_numbering_advice():
    found = rejects(REPEAT_TOOL, {"rep_0|item": {"src": "hdca", "id": "c1"}})

    assert _HINT not in found[0]["reason"]


def test_a_contiguous_run_of_instances_gets_no_numbering_advice():
    found = rejects(
        REPEAT_TOOL,
        {
            "rep_0|label": "first",
            "rep_1|label": "second",
            "rep_1|item": {"src": "hdca", "id": "c1"},
        },
    )

    assert _HINT not in found[0]["reason"]


def test_an_instance_past_a_gap_gets_the_advice():
    found = rejects(
        REPEAT_TOOL,
        {"rep_0|label": "first", "rep_2|item": {"src": "hdca", "id": "c1"}},
    )

    assert _HINT in found[0]["reason"]


def test_an_inner_repeat_is_numbered_within_its_outer_instance():
    reachable = rejects(
        NESTED_REPEAT,
        {
            "outer_0|inner_0|d": {"src": "hda", "id": "x"},
            "outer_0|inner_1|d": {"src": "hdca", "id": "c1"},
        },
    )
    stranded = rejects(
        NESTED_REPEAT,
        {
            "outer_0|inner_0|d": {"src": "hda", "id": "x"},
            "outer_1|inner_1|d": {"src": "hdca", "id": "c1"},
        },
    )

    assert _HINT not in reachable[0]["reason"]
    assert _HINT in stranded[0]["reason"]


def test_a_param_named_like_a_repeat_never_gets_numbering_advice():
    """q_1 is the whole parameter name, not instance 1 of a repeat named q."""
    found = rejects(PARAM_NAMED_LIKE_A_REPEAT, {"q_1": {"src": "hdca", "id": "c1"}})

    assert _HINT not in found[0]["reason"]


def test_a_repeat_with_a_minimum_reads_instances_below_it():
    """min="2" makes Galaxy create instances 0 and 1 itself, so rep_2 alone is read."""
    schema = {
        "id": "t",
        "inputs": [
            {
                "name": "rep",
                "type": "repeat",
                "min": 2,
                "default": 2,
                "max": 10,
                "inputs": [{"name": "item", "type": "data", "multiple": False}],
            }
        ],
    }

    found = rejects(schema, {"rep_2|item": {"src": "hdca", "id": "c1"}})

    assert _HINT not in found[0]["reason"]
    assert _HINT in rejects(schema, {"rep_3|item": {"src": "hdca", "id": "c1"}})[0]["reason"]


def test_a_junk_minimum_falls_back_to_zero():
    schema = {
        "id": "t",
        "inputs": [
            {
                "name": "rep",
                "type": "repeat",
                "min": None,
                "default": "2",
                "inputs": [{"name": "item", "type": "data", "multiple": False}],
            }
        ],
    }

    assert _HINT in rejects(schema, {"rep_2|item": {"src": "hdca", "id": "c1"}})[0]["reason"]


# ---------------------------------------------------------------------------
# Only the segments the schema says are repeat instances get generalized
# ---------------------------------------------------------------------------
#
# A section or a conditional may be declared `region_1` just as legitimately as an
# instance of a repeat named `region` is written `region_1`, and only the schema can
# tell them apart.

REPEAT_AROUND_INDEXED_SECTION = {
    "id": "t",
    "inputs": [
        {
            "name": "rep",
            "type": "repeat",
            "inputs": [
                {
                    "name": "region_1",
                    "type": "section",
                    "inputs": [{"name": "d", "type": "data", "multiple": False}],
                }
            ],
        }
    ],
}
REPEAT_AROUND_INDEXED_CONDITIONAL = {
    "id": "t",
    "inputs": [
        {
            "name": "rep",
            "type": "repeat",
            "inputs": [
                {
                    "name": "mode_2",
                    "type": "conditional",
                    "test_param": {"name": "how", "type": "select"},
                    "cases": [
                        {"when": "x", "inputs": [{"name": "d", "type": "data", "multiple": False}]}
                    ],
                }
            ],
        }
    ],
}


def test_a_section_named_like_an_instance_inside_a_repeat_resolves():
    verdict = check_tool_inputs(
        REPEAT_AROUND_INDEXED_SECTION, {"rep_0|region_1|d": {"src": "hda", "id": "x"}}
    )

    assert verdict == {"rejects": [], "warnings": []}


def test_a_section_named_like_an_instance_is_still_checked():
    found = rejects(REPEAT_AROUND_INDEXED_SECTION, {"rep_0|region_1|d": {"src": "hdca", "id": "c"}})

    assert [r["param"] for r in found] == ["rep_0|region_1|d"]


def test_a_conditional_named_like_an_instance_inside_a_repeat_resolves():
    verdict = check_tool_inputs(
        REPEAT_AROUND_INDEXED_CONDITIONAL, {"rep_1|mode_2|d": {"src": "hda", "id": "x"}}
    )

    assert verdict == {"rejects": [], "warnings": []}


def test_only_the_real_repeat_counts_toward_the_numbering_advice():
    """rep is the repeat; region_1 is a section name, so its 1 is not an index."""
    found = rejects(REPEAT_AROUND_INDEXED_SECTION, {"rep_0|region_1|d": {"src": "hdca", "id": "c"}})

    assert _HINT not in found[0]["reason"]
    assert (
        _HINT
        in rejects(REPEAT_AROUND_INDEXED_SECTION, {"rep_4|region_1|d": {"src": "hdca", "id": "c"}})[
            0
        ]["reason"]
    )


def test_a_nested_repeat_named_like_an_index_still_generalizes():
    """When the schema really does declare a repeat there, the wildcard still applies."""
    schema = {
        "id": "t",
        "inputs": [
            {
                "name": "rep",
                "type": "repeat",
                "inputs": [
                    {
                        "name": "region",
                        "type": "repeat",
                        "inputs": [{"name": "d", "type": "data", "multiple": False}],
                    }
                ],
            }
        ],
    }

    assert (
        check_tool_inputs(schema, {"rep_0|region_0|d": {"src": "hda", "id": "x"}})["warnings"] == []
    )
    assert _HINT in rejects(schema, {"rep_0|region_3|d": {"src": "hdca", "id": "c"}})[0]["reason"]


# ---------------------------------------------------------------------------
# A user-defined tool's stored representation nests its parameters differently
# ---------------------------------------------------------------------------
#
# io_details says "inputs", "test_param" and "cases"; a UserToolSource says
# "parameters", "test_parameter" and "whens" (galaxy/tool_util_models/
# yaml_parameters.py). run_user_tool checks against the representation, so reading
# only the first shape would leave every nested parameter of a user tool unchecked.

USER_TOOL = {
    "class": "GalaxyUserTool",
    "id": "my_filter",
    "version": "0.1.0",
    "container": "busybox",
    "inputs": [
        {"name": "top", "type": "data", "format": ["tabular"]},
        {
            "name": "sect",
            "type": "section",
            "parameters": [{"name": "f", "type": "data", "format": ["tabular"]}],
        },
        {
            "name": "rep",
            "type": "repeat",
            "min": 2,
            "parameters": [{"name": "item", "type": "data", "format": ["tabular"]}],
        },
        {
            "name": "cond",
            "type": "conditional",
            "test_parameter": {"name": "mode", "type": "boolean"},
            "whens": [
                {
                    "discriminator": True,
                    "parameters": [{"name": "c", "type": "data_collection"}],
                },
                {"discriminator": False, "parameters": []},
            ],
        },
    ],
}


def test_a_user_tool_representation_indexes_its_nested_params():
    assert sorted(index_tool_params(USER_TOOL)) == [
        "cond|c",
        "cond|mode",
        "rep_#|item",
        "sect|f",
        "top",
    ]


@pytest.mark.parametrize("key", ["top", "sect|f", "rep_0|item"])
def test_a_collection_into_a_user_tool_data_param_is_refused_wherever_it_sits(key):
    """A representation omits `multiple`, and YamlDataParameter defaults it to False."""
    out = rejects(USER_TOOL, {key: {"src": "hdca", "id": "c1"}})
    assert [r["param"] for r in out] == [key]
    assert "single dataset" in out[0]["reason"]


def test_a_user_tool_collection_param_still_refuses_a_dataset():
    schema = {
        "class": "GalaxyUserTool",
        "id": "my_filter",
        "inputs": [{"name": "coll", "type": "data_collection"}],
    }
    out = rejects(schema, {"coll": {"src": "hda", "id": "d1"}})

    assert [r["param"] for r in out] == ["coll"]
    assert "dataset collection" in out[0]["reason"]


def test_a_user_tool_param_under_an_unnamed_boolean_case_is_left_alone():
    """USER_TOOL's `c` sits under one branch of a boolean, which nothing here resolves."""
    assert rejects(USER_TOOL, {"cond|c": {"src": "hda", "id": "d1"}}) == []


def test_a_user_tool_nested_key_is_not_reported_as_unrecognised():
    verdict = check_tool_inputs(USER_TOOL, {"sect|f": {"src": "hda", "id": "d1"}})
    assert verdict == {"rejects": [], "warnings": []}


def test_a_user_tool_repeat_minimum_is_read_from_the_representation():
    """min: 2 means Galaxy already creates instances 0 and 1, so rep_1 needs no advice."""
    assert _HINT not in rejects(USER_TOOL, {"rep_1|item": {"src": "hdca", "id": "c1"}})[0]["reason"]
    assert _HINT in rejects(USER_TOOL, {"rep_3|item": {"src": "hdca", "id": "c1"}})[0]["reason"]


# ---------------------------------------------------------------------------
# Only the selected case of a conditional is ever read
# ---------------------------------------------------------------------------
#
# _populate_state_legacy resolves the test param, calls get_current_case, and then
# populates input.cases[current_case].inputs and nothing else. A value sitting under
# any other case is never looked at, so refusing it refuses a run that works.


def _one_sided_conditional(selector, cases=None):
    return {
        "id": "t",
        "version": "1.0.0",
        "inputs": [
            {
                "name": "c",
                "type": "conditional",
                "test_param": selector,
                "cases": cases
                or [
                    {"value": "a", "inputs": [{"name": "d", "type": "data", "multiple": False}]},
                    {"value": "b", "inputs": []},
                ],
            }
        ],
    }


HDCA = {"src": "hdca", "id": "c1"}
PICKS_A = {"name": "mode", "type": "select", "value": "a"}
NO_DEFAULT = {"name": "mode", "type": "select"}


def test_a_value_under_an_unselected_case_is_not_refused():
    """Case b declares no `d`, so Galaxy never reads `c|d` and neither may we."""
    schema = _one_sided_conditional(PICKS_A)

    assert rejects(schema, {"c|mode": "b", "c|d": HDCA}) == []


def test_the_selected_case_is_still_checked():
    schema = _one_sided_conditional(PICKS_A)

    assert [r["param"] for r in rejects(schema, {"c|mode": "a", "c|d": HDCA})] == ["c|d"]


def test_a_selector_value_matching_no_case_names_nothing():
    """It is not one of the declared values, so it settles nothing about the branch."""
    schema = _one_sided_conditional(PICKS_A)

    assert rejects(schema, {"c|mode": "not-a-case", "c|d": HDCA}) == []


def test_the_schemas_own_default_is_never_read_as_the_selected_case():
    """A stored default is not the runtime one, and a response may omit it entirely."""
    picks_b = {"name": "mode", "type": "select", "value": "b"}

    assert rejects(_one_sided_conditional(PICKS_A), {"c|d": HDCA}) == []
    assert rejects(_one_sided_conditional(picks_b), {"c|d": HDCA}) == []


def test_a_selected_option_is_not_read_as_a_default_either():
    selector = {"name": "mode", "type": "select", "options": [["A", "a", False], ["B", "b", True]]}

    assert rejects(_one_sided_conditional(selector), {"c|d": HDCA}) == []


def test_without_a_named_case_a_value_is_refused_only_if_every_case_declares_the_key():
    """A case that never mentions the key is a case where Galaxy ignores it."""
    both_wrong = [
        {"value": "a", "inputs": [{"name": "d", "type": "data", "multiple": False}]},
        {"value": "b", "inputs": [{"name": "d", "type": "data", "multiple": False}]},
    ]
    disagreeing = [
        {"value": "a", "inputs": [{"name": "d", "type": "data", "multiple": False}]},
        {"value": "b", "inputs": [{"name": "d", "type": "data", "multiple": True}]},
    ]

    # Wrong whichever case runs, so it can be said without knowing which one does.
    assert [
        r["param"] for r in rejects(_one_sided_conditional(NO_DEFAULT, both_wrong), {"c|d": HDCA})
    ] == ["c|d"]
    # The cases disagree about it, so nothing can be said.
    assert rejects(_one_sided_conditional(NO_DEFAULT, disagreeing), {"c|d": HDCA}) == []
    # One case does not declare it at all, which is a case where it is ignored.
    assert rejects(_one_sided_conditional(NO_DEFAULT), {"c|d": HDCA}) == []
    # Naming the case resolves all three the same way.
    assert (
        rejects(_one_sided_conditional(NO_DEFAULT, disagreeing), {"c|mode": "b", "c|d": HDCA}) == []
    )
    assert [
        r["param"]
        for r in rejects(
            _one_sided_conditional(NO_DEFAULT, disagreeing), {"c|mode": "a", "c|d": HDCA}
        )
    ] == ["c|d"]


def test_a_boolean_selector_names_nothing_even_when_it_looks_like_a_match():
    """from_json puts the value through string_as_bool first, so "enabled" is False."""
    selector = {"name": "flag", "type": "boolean", "truevalue": "enabled", "falsevalue": "disabled"}
    cases = [
        {"value": "enabled", "inputs": [{"name": "d", "type": "data", "multiple": False}]},
        {"value": "disabled", "inputs": [{"name": "d", "type": "data", "multiple": True}]},
    ]
    schema = _one_sided_conditional(selector, cases)

    # The literal string matches the "enabled" case and Galaxy would still take the
    # other one, so neither case may be picked here.
    assert rejects(schema, {"c|flag": "enabled", "c|d": HDCA}) == []
    assert rejects(schema, {"c|flag": True, "c|d": HDCA}) == []
    assert rejects(schema, {"c|flag": False, "c|d": HDCA}) == []


def test_a_boolean_selector_cannot_rescue_a_value_wrong_under_every_case():
    """Not knowing the case is not a reason to ignore a value no case accepts."""
    selector = {"name": "flag", "type": "boolean", "truevalue": "enabled", "falsevalue": "disabled"}
    cases = [
        {"value": "enabled", "inputs": [{"name": "d", "type": "data", "multiple": False}]},
        {"value": "disabled", "inputs": [{"name": "d", "type": "data", "multiple": False}]},
    ]

    assert [
        r["param"] for r in rejects(_one_sided_conditional(selector, cases), {"c|d": HDCA})
    ] == ["c|d"]


def test_a_conditional_inside_a_repeat_narrows_per_instance():
    """The selector key carries the instance index, so each instance decides alone."""
    schema = {
        "id": "t",
        "inputs": [
            {
                "name": "rep",
                "type": "repeat",
                "inputs": [
                    {
                        "name": "c",
                        "type": "conditional",
                        "test_param": PICKS_A,
                        "cases": [
                            {
                                "value": "a",
                                "inputs": [{"name": "d", "type": "data", "multiple": False}],
                            },
                            {"value": "b", "inputs": []},
                        ],
                    }
                ],
            }
        ],
    }
    inputs = {
        "rep_0|c|mode": "b",
        "rep_0|c|d": HDCA,
        "rep_1|c|mode": "a",
        "rep_1|c|d": HDCA,
    }

    assert [r["param"] for r in rejects(schema, inputs)] == ["rep_1|c|d"]


def test_a_user_tool_select_discriminator_still_names_a_case():
    """whens/discriminator is the same statement as cases/value, and reads the same."""
    schema = {
        "class": "GalaxyUserTool",
        "id": "my_filter",
        "inputs": [
            {
                "name": "c",
                "type": "conditional",
                "test_parameter": {"name": "mode", "type": "select"},
                "whens": [
                    {"discriminator": "one", "parameters": [{"name": "d", "type": "data"}]},
                    {"discriminator": "two", "parameters": []},
                ],
            }
        ],
    }

    assert rejects(schema, {"c|mode": "two", "c|d": HDCA}) == []
    assert [r["param"] for r in rejects(schema, {"c|mode": "one", "c|d": HDCA})] == ["c|d"]


def test_a_user_tool_boolean_default_is_never_assumed():
    """boolean_is_checked reads `checked`, not the `value` a representation carries."""
    schema = {
        "class": "GalaxyUserTool",
        "id": "my_filter",
        "inputs": [
            {
                "name": "c",
                "type": "conditional",
                "test_parameter": {"name": "flag", "type": "boolean", "value": True},
                "whens": [
                    {"discriminator": True, "parameters": [{"name": "d", "type": "data"}]},
                    {
                        "discriminator": False,
                        "parameters": [{"name": "d", "type": "data", "multiple": True}],
                    },
                ],
            }
        ],
    }

    assert rejects(schema, {"c|d": HDCA}) == []


def test_a_representation_that_omits_its_boolean_default_is_not_guessed_at():
    """The GET is response_model_exclude_defaults, so the field to guess from is gone."""
    schema = {
        "class": "GalaxyUserTool",
        "id": "my_filter",
        "inputs": [
            {
                "name": "c",
                "type": "conditional",
                "test_parameter": {"name": "flag", "type": "boolean"},
                "whens": [
                    {"discriminator": True, "parameters": [{"name": "d", "type": "data"}]},
                    {"discriminator": False, "parameters": []},
                ],
            }
        ],
    }

    assert rejects(schema, {"c|d": HDCA}) == []


def test_a_key_only_the_unselected_case_declares_is_not_called_unrecognised():
    """It is a real parameter of the tool; Galaxy just will not read it this run."""
    verdict = check_tool_inputs(_one_sided_conditional(PICKS_A), {"c|mode": "b", "c|d": HDCA})

    assert verdict == {"rejects": [], "warnings": []}


# ---------------------------------------------------------------------------
# A schema the checker cannot read is unchecked, not all-clear
# ---------------------------------------------------------------------------


def test_a_schema_the_checker_chokes_on_is_reported_not_swallowed():
    """An empty verdict and "could not read it" must not look the same to a caller."""
    broken = {"id": "t", "inputs": [{"name": "c", "type": "conditional", "whens": 1}]}

    with pytest.raises(ToolInputsUncheckableError) as exc:
        check_tool_inputs(broken, {"x": HDCA})

    assert "TypeError" in str(exc.value)


# ---------------------------------------------------------------------------
# Nested conditionals with nothing naming a case
# ---------------------------------------------------------------------------
#
# The every-case claim needs the whole conditional in view. Counting two nested
# conditionals separately says a key is covered when it is declared under a/x, a/y
# and b/x, and a run that resolves to b/y never reads it.


def _nested_conditionals(p_under_by):
    def inner(with_p):
        return {
            "name": "c2",
            "type": "conditional",
            "test_param": {"name": "s2", "type": "select"},
            "cases": [
                {"value": "x", "inputs": [{"name": "P", "type": "data", "multiple": False}]},
                {
                    "value": "y",
                    "inputs": [{"name": "P", "type": "data", "multiple": False}] if with_p else [],
                },
            ],
        }

    return {
        "id": "t",
        "version": "1.0.0",
        "inputs": [
            {
                "name": "c1",
                "type": "conditional",
                "test_param": {"name": "s1", "type": "select"},
                "cases": [
                    {"value": "a", "inputs": [inner(True)]},
                    {"value": "b", "inputs": [inner(p_under_by)]},
                ],
            }
        ],
    }


def test_a_combination_that_never_declares_the_key_is_not_covered():
    """Declared under a/x, a/y and b/x: a run resolving to b/y never reads it."""
    assert rejects(_nested_conditionals(p_under_by=False), {"c1|c2|P": HDCA}) == []


def test_two_unnamed_conditionals_make_no_claim_even_when_every_combination_declares_it():
    """The combinations are what would have to be enumerated, so nothing is said."""
    assert rejects(_nested_conditionals(p_under_by=True), {"c1|c2|P": HDCA}) == []


def test_naming_the_outer_case_leaves_one_conditional_to_cover():
    """With the outer settled, the inner is a single conditional again."""
    covered = _nested_conditionals(p_under_by=True)

    assert [r["param"] for r in rejects(covered, {"c1|s1": "a", "c1|c2|P": HDCA})] == ["c1|c2|P"]
    # Under b the key is declared in x but not y, so the inner is not covered.
    assert rejects(_nested_conditionals(p_under_by=False), {"c1|s1": "b", "c1|c2|P": HDCA}) == []


def test_naming_both_cases_still_judges_the_value():
    schema = _nested_conditionals(p_under_by=True)
    inputs = {"c1|s1": "b", "c1|c2|s2": "y", "c1|c2|P": HDCA}

    assert [r["param"] for r in rejects(schema, inputs)] == ["c1|c2|P"]
