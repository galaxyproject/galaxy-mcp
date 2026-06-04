import json as _json
from pathlib import Path

from galaxy_mcp.workflow_inputs import (
    build_workflow_input_template,
    find_legacy_warnings,
    normalize_ga_steps,
    normalize_run_model,
    subtype_satisfies,
    validate_inputs,
)

# Minimal slice of /api/datatypes/types_and_mapping
MAPPING = {
    "ext_to_class_name": {
        "bam": "galaxy.datatypes.binary.Bam",
        "tabular": "galaxy.datatypes.tabular.Tabular",
        "bed": "galaxy.datatypes.interval.Bed",
        "fastqsanger": "galaxy.datatypes.sequence.FastqSanger",
    },
    # class -> its ancestor classes (membership-tested; dict per live API)
    "class_to_classes": {
        "galaxy.datatypes.binary.Bam": {"galaxy.datatypes.binary.Bam": True},
        "galaxy.datatypes.interval.Bed": {
            "galaxy.datatypes.interval.Bed": True,
            "galaxy.datatypes.tabular.Tabular": True,
        },
        "galaxy.datatypes.sequence.FastqSanger": {
            "galaxy.datatypes.sequence.FastqSanger": True,
        },
    },
}


def test_subtype_satisfies_empty_accepts_anything():
    assert subtype_satisfies("bam", [], MAPPING) is True


def test_subtype_satisfies_exact_match():
    assert subtype_satisfies("bam", ["bam"], MAPPING) is True


def test_subtype_satisfies_subclass():
    # bed is-a tabular
    assert subtype_satisfies("bed", ["tabular"], MAPPING) is True


def test_subtype_satisfies_rejects_unrelated():
    # bam is not fastqsanger/tabular
    assert subtype_satisfies("bam", ["fastqsanger", "tabular"], MAPPING) is False


def test_subtype_satisfies_unknown_supplied_ext_is_permissive():
    # unknown extension -> cannot prove a mismatch -> do not reject
    assert subtype_satisfies("mystery", ["bam"], MAPPING) is True


# ---------------------------------------------------------------------------
# Task 3: .ga / manifest normalizer
# ---------------------------------------------------------------------------

GA_DEF = {
    "steps": {
        "0": {  # restricted data input
            "type": "data_input",
            "label": "barcodes",
            "uuid": "u0",
            "tool_state": '{"optional": false, "format": ["tabular"], "tag": ""}',
        },
        "1": {  # unrestricted data input (format absent)
            "type": "data_input",
            "label": None,
            "uuid": "u1",
            "tool_state": '{"optional": false, "tag": ""}',
        },
        "2": {  # collection input
            "type": "data_collection_input",
            "label": "reads",
            "uuid": "u2",
            "tool_state": (
                '{"optional": false, "format": ["fastqsanger"], "collection_type": "list:paired"}'
            ),
        },
        "3": {  # scalar parameter input
            "type": "parameter_input",
            "label": "strandedness",
            "uuid": "u3",
            "tool_state": '{"parameter_type": "text", "optional": true}',
        },
        "4": {"type": "tool", "label": "FastQC", "tool_state": "{}"},  # ignored
    }
}


def test_normalize_ga_steps_extracts_only_input_slots():
    slots = normalize_ga_steps(GA_DEF)
    assert [s["step_index"] for s in slots] == [0, 1, 2, 3]


def test_normalize_ga_steps_data_input_restricted():
    s = normalize_ga_steps(GA_DEF)[0]
    assert s == {
        "step_index": 0,
        "step_uuid": "u0",
        "label": "barcodes",
        "input_type": "data",
        "src": "hda",
        "accepted_formats": ["tabular"],
        "collection_type": None,
        "parameter_type": None,
        "optional": False,
    }


def test_normalize_ga_steps_unrestricted_data_input_has_empty_formats_and_fallback_label():
    s = normalize_ga_steps(GA_DEF)[1]
    assert s["accepted_formats"] == []
    assert s["label"] == "Input dataset (step 1)"


def test_normalize_ga_steps_collection_input():
    s = normalize_ga_steps(GA_DEF)[2]
    assert s["input_type"] == "data_collection"
    assert s["src"] == "hdca"
    assert s["collection_type"] == "list:paired"
    assert s["accepted_formats"] == ["fastqsanger"]


def test_normalize_ga_steps_parameter_input():
    s = normalize_ga_steps(GA_DEF)[3]
    assert s["input_type"] == "parameter"
    assert s["src"] is None
    assert s["parameter_type"] == "text"
    assert s["optional"] is True


# ---------------------------------------------------------------------------
# Task 4: style=run normalizer (fixture-driven)
# ---------------------------------------------------------------------------

_FIXTURE = Path(__file__).parent / "testdata" / "wf_style_run_rnaseq.json"


def test_normalize_run_model_returns_slot_contract():
    run_dict = _json.loads(_FIXTURE.read_text())
    slots = normalize_run_model(run_dict)
    assert slots, "expected at least one input slot"
    for s in slots:
        assert set(s) == {
            "step_index",
            "step_uuid",
            "label",
            "input_type",
            "src",
            "accepted_formats",
            "collection_type",
            "parameter_type",
            "optional",
        }
        assert s["input_type"] in {"data", "data_collection", "parameter"}
        assert isinstance(s["accepted_formats"], list)
    # step indices are ints and unique
    idx = [s["step_index"] for s in slots]
    assert idx == sorted(set(idx))


# ---------------------------------------------------------------------------
# Task 5: legacy RuntimeValue scanner
# ---------------------------------------------------------------------------

GA_LEGACY = {
    "steps": {
        "0": {
            "type": "parameter_input",
            "label": "p",
            "tool_state": '{"parameter_type":"text"}',
        },
        "1": {  # tool step with an unconnected RuntimeValue -> legacy
            "type": "tool",
            "label": "Cut",
            "tool_state": (
                '{"col": {"__class__": "RuntimeValue"}, "ref": {"__class__": "ConnectedValue"}}'
            ),
        },
        "2": {"type": "tool", "label": "Clean", "tool_state": '{"opt": "x"}'},
    }
}


def test_find_legacy_warnings_flags_runtimevalue_not_parameter_input():
    warns = find_legacy_warnings(GA_LEGACY)
    joined = " ".join(w["message"] for w in warns)
    assert "RuntimeValue" in joined
    assert "Cut" in joined  # the offending tool step is named
    assert "parameter_input" not in joined  # bare parameter_input is NOT flagged


def test_find_legacy_warnings_clean_workflow_is_empty():
    assert find_legacy_warnings({"steps": {"0": {"type": "tool", "tool_state": '{"a":1}'}}}) == []


# ---------------------------------------------------------------------------
# Task 6: three-tier validator
# ---------------------------------------------------------------------------

SLOTS = [
    {
        "step_index": 0,
        "step_uuid": "u0",
        "label": "barcodes",
        "input_type": "data",
        "src": "hda",
        "accepted_formats": ["tabular"],
        "collection_type": None,
        "parameter_type": None,
        "optional": False,
    },
    {
        "step_index": 1,
        "step_uuid": "u1",
        "label": "reads",
        "input_type": "data_collection",
        "src": "hdca",
        "accepted_formats": ["fastqsanger"],
        "collection_type": "list:paired",
        "parameter_type": None,
        "optional": False,
    },
    {
        "step_index": 2,
        "step_uuid": "u2",
        "label": "anyfile",
        "input_type": "data",
        "src": "hda",
        "accepted_formats": [],
        "collection_type": None,
        "parameter_type": None,
        "optional": True,
    },
]
MAP = MAPPING  # from Task 1


def test_validate_hard_rejects_wrong_datatype():
    supplied = {"0": {"src": "hda", "id": "d1", "ext": "bam"}}  # bam into a tabular slot
    res = validate_inputs(SLOTS, supplied, MAP)
    assert any(r["step_index"] == 0 for r in res["rejects"])


def test_validate_accepts_subtype():
    supplied = {"0": {"src": "hda", "id": "d1", "ext": "bed"}}  # bed is-a tabular
    res = validate_inputs(SLOTS, supplied, MAP)
    assert res["rejects"] == []


def test_validate_hard_rejects_wrong_src_kind():
    supplied = {"0": {"src": "hdca", "id": "c1", "collection_type": "list"}}
    res = validate_inputs(SLOTS, supplied, MAP)
    assert any(r["step_index"] == 0 and "collection" in r["reason"].lower() for r in res["rejects"])


def test_validate_hard_rejects_wrong_collection_type():
    supplied = {
        "1": {
            "src": "hdca",
            "id": "c1",
            "collection_type": "list",
            "element_extensions": ["fastqsanger"],
        }
    }
    res = validate_inputs(SLOTS, supplied, MAP)
    assert any(r["step_index"] == 1 for r in res["rejects"])


def test_validate_generic_data_slot_does_not_reject():
    supplied = {"2": {"src": "hda", "id": "d9", "ext": "bam"}}  # slot accepts any
    res = validate_inputs(SLOTS, supplied, MAP)
    assert res["rejects"] == []


def test_validate_unknown_step_index_is_warned_not_rejected():
    supplied = {"99": {"src": "hda", "id": "dx", "ext": "bam"}}
    res = validate_inputs(SLOTS, supplied, MAP)
    assert res["rejects"] == []
    assert any("99" in w["message"] for w in res["warnings"])


# ---------------------------------------------------------------------------
# Task 7: template builder
# ---------------------------------------------------------------------------


def test_build_template_skeleton_and_slots():
    tmpl = build_workflow_input_template(SLOTS, warnings=[{"kind": "x", "message": "m"}])
    assert tmpl["inputs_by"] == "step_index|step_uuid"
    # placeholders keyed by step_index
    assert tmpl["inputs_template"]["0"] == {"src": "hda", "id": "<dataset_id>"}
    assert tmpl["inputs_template"]["1"] == {"src": "hdca", "id": "<collection_id>"}
    # slot summary carries the human-facing constraints
    barcodes = next(s for s in tmpl["slots"] if s["step_index"] == 0)
    assert barcodes["accepted_formats"] == ["tabular"]
    assert tmpl["warnings"][0]["message"] == "m"


# ---------------------------------------------------------------------------
# Fix 1: _collection_type_compatible -- segment comparison, not raw suffix
# ---------------------------------------------------------------------------

from galaxy_mcp.workflow_inputs import _collection_type_compatible  # noqa: E402


def test_collection_type_compatible_list_does_not_satisfy_paired():
    # "list" does not end with ":paired"; the bare-suffix check wrongly passed before the fix
    assert _collection_type_compatible("list", "paired") is False


def test_collection_type_compatible_raw_suffix_regression():
    # "list" ends with "st" as a raw string -- must be False
    assert _collection_type_compatible("list", "st") is False


def test_collection_type_compatible_map_over_list_paired():
    # list:paired can feed a 'paired' slot via map-over
    assert _collection_type_compatible("list:paired", "paired") is True


def test_collection_type_compatible_exact():
    assert _collection_type_compatible("paired", "paired") is True


def test_collection_type_compatible_none_required():
    assert _collection_type_compatible(None, "list") is True


def test_collection_type_compatible_none_supplied():
    assert _collection_type_compatible("list", None) is True


# ---------------------------------------------------------------------------
# Fix 2: normalize_run_model -- missing/non-numeric step index
# ---------------------------------------------------------------------------


def test_normalize_run_model_skips_steps_with_no_index():
    """Steps missing all index keys AND steps with a hash-string id must not raise."""
    run_dict = {
        "steps": [
            # well-formed input step
            {
                "step_type": "data_input",
                "step_index": 0,
                "uuid": "u0",
                "step_label": "mydata",
                "inputs": [{"extensions": ["bam"], "optional": False}],
            },
            # no step_index, no order_index, no id at all
            {
                "step_type": "data_input",
                "uuid": "u-bad",
                "step_label": "orphan",
                "inputs": [{}],
            },
            # id is a hash string (Galaxy sometimes uses UUIDs as step ids)
            {
                "step_type": "data_input",
                "id": "abc123def",
                "uuid": "u-hash",
                "step_label": "hashstep",
                "inputs": [{}],
            },
        ]
    }
    slots = normalize_run_model(run_dict)
    # only the well-formed slot survives; the two bad ones are skipped
    assert len(slots) == 1
    assert slots[0]["step_index"] == 0


# ---------------------------------------------------------------------------
# Fix 3: normalize_ga_steps / find_legacy_warnings -- non-numeric step keys
# ---------------------------------------------------------------------------

GA_DEF_NONNUMERIC = {
    "steps": {
        "0": {
            "type": "data_input",
            "label": "good_step",
            "uuid": "u0",
            "tool_state": '{"optional": false, "format": ["bam"]}',
        },
        # non-numeric key -- should be skipped gracefully
        "x": {
            "type": "data_input",
            "label": "bad_key_step",
            "uuid": "ux",
            "tool_state": '{"optional": false}',
        },
    }
}

GA_LEGACY_NONNUMERIC = {
    "steps": {
        "0": {
            "type": "tool",
            "label": "NormalTool",
            "tool_state": '{"col": {"__class__": "RuntimeValue"}}',
        },
        "x": {
            "type": "tool",
            "label": "WeirdKeyTool",
            "tool_state": '{"val": 1}',
        },
    }
}


def test_normalize_ga_steps_nonnumeric_key_does_not_raise():
    slots = normalize_ga_steps(GA_DEF_NONNUMERIC)
    # the numeric step survives; the 'x' key is skipped
    assert len(slots) == 1
    assert slots[0]["step_index"] == 0


def test_find_legacy_warnings_nonnumeric_key_does_not_raise():
    # should not raise; the tool step with key "0" is flagged; "x" key is handled gracefully
    warns = find_legacy_warnings(GA_LEGACY_NONNUMERIC)
    assert any("NormalTool" in w["message"] for w in warns)


# ---------------------------------------------------------------------------
# Fix 4: format / extensions as a bare string
# ---------------------------------------------------------------------------

GA_DEF_FORMAT_STRING = {
    "steps": {
        "0": {
            "type": "data_input",
            "label": "bamfile",
            "uuid": "u0",
            "tool_state": '{"optional": false, "format": "bam"}',
        },
    }
}


def test_normalize_ga_steps_bare_string_format_is_single_element_list():
    slots = normalize_ga_steps(GA_DEF_FORMAT_STRING)
    assert slots[0]["accepted_formats"] == ["bam"]


def test_normalize_run_model_bare_string_extensions_is_single_element_list():
    run_dict = {
        "steps": [
            {
                "step_type": "data_input",
                "step_index": 0,
                "uuid": "u0",
                "step_label": "bamfile",
                "inputs": [{"extensions": "bam", "optional": False}],
            }
        ]
    }
    slots = normalize_run_model(run_dict)
    assert slots[0]["accepted_formats"] == ["bam"]


# ---------------------------------------------------------------------------
# Fix 5: duplicate step_index produces a warning
# ---------------------------------------------------------------------------


def test_validate_duplicate_step_index_warns():
    dup_slots = [
        {
            "step_index": 0,
            "step_uuid": "ua",
            "label": "first",
            "input_type": "data",
            "src": "hda",
            "accepted_formats": [],
            "collection_type": None,
            "parameter_type": None,
            "optional": True,
        },
        {
            "step_index": 0,
            "step_uuid": "ub",
            "label": "second",
            "input_type": "data",
            "src": "hda",
            "accepted_formats": [],
            "collection_type": None,
            "parameter_type": None,
            "optional": True,
        },
    ]
    res = validate_inputs(dup_slots, {}, MAPPING)
    assert any("0" in w["message"] for w in res["warnings"])
