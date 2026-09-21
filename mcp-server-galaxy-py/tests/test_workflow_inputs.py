import json as _json
from pathlib import Path

import pytest

from galaxy_mcp.workflow_inputs import (
    _clean_readme_summary,
    _collection_type_compatible,
    build_guide,
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
        "acceptable_extensions": [],
        "collection_type": None,
        "parameter_type": None,
        "optional": False,
        "options": [],
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
            "acceptable_extensions",
            "collection_type",
            "parameter_type",
            "optional",
            "options",
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
# Shaped like something encode_id would produce: 16 hex digits.
LIB_ID = "f2db41e1fa331b3e"


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


# ---------------------------------------------------------------------------
# Converter parity: acceptable_extensions membership vs subclass closure
# ---------------------------------------------------------------------------


def test_validate_uses_acceptable_extensions_for_converter_parity():
    slot = {
        "step_index": 0,
        "step_uuid": None,
        "label": "counts",
        "input_type": "data",
        "src": "hda",
        "accepted_formats": ["tabular"],
        "acceptable_extensions": ["tabular", "tsv", "csv"],  # Galaxy's converter-aware set
        "collection_type": None,
        "parameter_type": None,
        "optional": False,
    }
    # csv is NOT a Tabular subclass, but Galaxy lists it as acceptable -> must NOT reject
    ok = validate_inputs([slot], {"0": {"src": "hda", "id": "d1", "ext": "csv"}}, MAPPING)
    assert ok["rejects"] == []
    # bam is not in Galaxy's accept-set -> provable reject
    bad = validate_inputs([slot], {"0": {"src": "hda", "id": "d2", "ext": "bam"}}, MAPPING)
    assert any(r["step_index"] == 0 for r in bad["rejects"])


def test_validate_falls_back_to_subtype_closure_without_acceptable_extensions():
    slot = {
        "step_index": 0,
        "step_uuid": None,
        "label": "x",
        "input_type": "data",
        "src": "hda",
        "accepted_formats": ["tabular"],
        "acceptable_extensions": [],
        "collection_type": None,
        "parameter_type": None,
        "optional": False,
    }
    # bed IS-A tabular per MAPPING -> accepted via closure
    assert (
        validate_inputs([slot], {"0": {"src": "hda", "id": "d1", "ext": "bed"}}, MAPPING)["rejects"]
        == []
    )
    # bam is not -> rejected via closure
    assert any(
        r["step_index"] == 0
        for r in validate_inputs([slot], {"0": {"src": "hda", "id": "d2", "ext": "bam"}}, MAPPING)[
            "rejects"
        ]
    )


def test_build_template_omits_acceptable_extensions_from_display_slots():
    slots = [
        {
            "step_index": 0,
            "step_uuid": None,
            "label": "x",
            "input_type": "data",
            "src": "hda",
            "accepted_formats": ["txt"],
            "acceptable_extensions": ["txt", "csv", "tsv"],
            "collection_type": None,
            "parameter_type": None,
            "optional": False,
        }
    ]
    tmpl = build_workflow_input_template(slots)
    assert "acceptable_extensions" not in tmpl["slots"][0]
    assert tmpl["slots"][0]["accepted_formats"] == ["txt"]


# ---------------------------------------------------------------------------
# Task 1 (run-guide): _clean_readme_summary moved to the pure module
# ---------------------------------------------------------------------------


def test_clean_readme_summary_strips_headers_and_truncates():
    md = "# Title\n\nThis workflow does X.\n## Details\nThen Y."
    out = _clean_readme_summary(md, max_length=40)
    assert "#" not in out
    assert out.startswith("This workflow does X.")
    assert len(out) <= 40


def test_clean_readme_summary_empty():
    assert _clean_readme_summary("") == ""


# ---------------------------------------------------------------------------
# Task 2 (run-guide): options field on the slot contract
# ---------------------------------------------------------------------------


def test_ga_enumerated_restrictions_become_options():
    ga = {
        "steps": {
            "0": {
                "type": "parameter_input",
                "label": "Strand",
                "tool_state": '{"parameter_type": "text", "restrictions": ["fwd", "rev"]}',
            }
        }
    }
    s = normalize_ga_steps(ga)[0]
    assert s["options"] == [{"label": "fwd", "value": "fwd"}, {"label": "rev", "value": "rev"}]


def test_ga_data_input_has_empty_options():
    ga = {"steps": {"0": {"type": "data_input", "label": "in", "tool_state": "{}"}}}
    assert normalize_ga_steps(ga)[0]["options"] == []


def test_run_model_options_from_triples():
    # the real style=run shape: inputs[0].options is a list of [label, value, selected]
    run = {
        "steps": [
            {
                "step_type": "parameter_input",
                "step_index": 0,
                "step_label": "Strand",
                "inputs": [{"options": [["forward", "fwd", False], ["reverse", "rev", True]]}],
            }
        ]
    }
    s = normalize_run_model(run)[0]
    assert s["options"] == [
        {"label": "forward", "value": "fwd"},
        {"label": "reverse", "value": "rev"},
    ]


def test_run_model_real_fixture_has_strandedness_options():
    fixture = Path(__file__).parent / "testdata" / "wf_style_run_rnaseq.json"
    slots = normalize_run_model(_json.loads(fixture.read_text()))
    strand = next(s for s in slots if s["label"] == "Strandedness")
    assert {o["value"] for o in strand["options"]} == {
        "stranded - forward",
        "stranded - reverse",
        "unstranded",
    }
    ref = next(s for s in slots if s["label"] == "Reference genome")
    assert ref["options"]
    assert all("value" in o for o in ref["options"])


# ---------------------------------------------------------------------------
# Task 3 (run-guide): build_guide assembles the model-facing run guide
# ---------------------------------------------------------------------------

WF_SHOW = {
    "version": 12,
    "annotation": "RNA-seq for paired-end data",
    "readme": "# RNA-Seq\n\nThis runs adapter trimming then alignment then quantification. " * 10,
    "help": "",
    "source_metadata": {
        "trs_tool_id": "#workflow/.../rnaseq-pe/main",
        "trs_url": "https://dockstore/x",
    },
}
RUN_MODEL = {"has_upgrade_messages": False, "step_version_changes": []}


def test_build_guide_summary_is_short_by_default():
    g = build_guide(WF_SHOW, RUN_MODEL, verbose=False)
    assert len(g["summary"]) <= 300
    assert g["annotation"] == "RNA-seq for paired-end data"
    assert g["provenance"]["version"] == 12
    assert g["provenance"]["source"]["trs_id"] == "#workflow/.../rnaseq-pe/main"
    assert g["provenance"]["freshness"] == {
        "has_upgrade_messages": False,
        "step_version_changes": [],
    }


def test_build_guide_verbose_returns_full_readme():
    g = build_guide(WF_SHOW, RUN_MODEL, verbose=True)
    assert len(g["summary"]) > 300  # full readme, not truncated


def test_build_guide_without_run_model_omits_freshness_and_adds_note():
    g = build_guide(WF_SHOW, None, verbose=False)
    assert "freshness" not in g["provenance"]
    assert any("history_id" in n for n in g.get("notes", []))


def test_build_guide_falls_back_to_annotation_when_no_readme():
    g = build_guide(
        {"version": 1, "annotation": "Just an annotation", "readme": "", "help": ""},
        None,
        verbose=False,
    )
    assert g["summary"] == "Just an annotation"


# ---------------------------------------------------------------------------
# Task 4 (run-guide): option-capping + guide in build_workflow_input_template
# ---------------------------------------------------------------------------


def _param_slot(options):
    return {
        "step_index": 0,
        "step_uuid": None,
        "label": "Genome",
        "input_type": "parameter",
        "src": None,
        "accepted_formats": [],
        "acceptable_extensions": [],
        "collection_type": None,
        "parameter_type": "text",
        "optional": False,
        "options": options,
    }


def test_template_inlines_small_option_sets():
    opts = [{"label": f"g{i}", "value": str(i)} for i in range(5)]
    t = build_workflow_input_template([_param_slot(opts)])
    s = t["slots"][0]
    assert s["options"] == opts
    assert s["option_count"] == 5
    assert "options_note" not in s


def test_template_caps_large_option_sets_unless_verbose():
    opts = [{"label": f"g{i}", "value": str(i)} for i in range(100)]
    s = build_workflow_input_template([_param_slot(opts)])["slots"][0]
    assert len(s["options"]) == 15
    assert s["option_count"] == 100
    assert "options_note" in s
    # verbose returns the full list
    sv = build_workflow_input_template([_param_slot(opts)], verbose=True)["slots"][0]
    assert len(sv["options"]) == 100
    assert "options_note" not in sv


def test_template_drops_empty_options_and_strips_acceptable_extensions():
    slot = _param_slot([])
    slot["acceptable_extensions"] = ["a", "b"]
    s = build_workflow_input_template([slot])["slots"][0]
    assert "options" not in s  # empty -> dropped from display
    assert "option_count" not in s
    assert "acceptable_extensions" not in s  # still stripped (from #55)


def test_template_includes_guide_when_provided():
    g = {"summary": "x", "provenance": {"version": 1}}
    t = build_workflow_input_template([_param_slot([])], guide=g)
    assert t["guide"] == g


# ---------------------------------------------------------------------------
# Fix 1 (new): options must only appear on parameter inputs
# ---------------------------------------------------------------------------


def test_ga_data_input_ignores_stray_restrictions():
    ga = {
        "steps": {
            "0": {"type": "data_input", "label": "in", "tool_state": '{"restrictions": ["x", "y"]}'}
        }
    }
    assert normalize_ga_steps(ga)[0]["options"] == []


def test_run_data_input_ignores_list_options():
    run = {
        "steps": [
            {
                "step_type": "data_input",
                "step_index": 0,
                "step_label": "in",
                "inputs": [{"options": [["a", "a", False]]}],
            }
        ]
    }
    assert normalize_run_model(run)[0]["options"] == []


# ---------------------------------------------------------------------------
# Fix 2 (new): build_guide summary must fall through a headers-only readme
# ---------------------------------------------------------------------------


def test_build_guide_skips_headers_only_readme_for_help():
    g = build_guide(
        {
            "version": 1,
            "annotation": "ann",
            "readme": "# Title\n## Sub",
            "help": "Real help text here.",
        },
        None,
        verbose=False,
    )
    assert "Real help text" in g["summary"]


def test_build_guide_headers_only_readme_no_help_uses_annotation():
    g = build_guide(
        {"version": 1, "annotation": "Just an annotation", "readme": "# Only headers", "help": ""},
        None,
        verbose=False,
    )
    assert g["summary"] == "Just an annotation"


# ---------------------------------------------------------------------------
# Fix 3 (new): option label/value must be string-coerced
# ---------------------------------------------------------------------------


def test_options_values_are_stringified():
    run = {
        "steps": [
            {
                "step_type": "parameter_input",
                "step_index": 0,
                "step_label": "p",
                "inputs": [{"options": [["one", 1, False], ["two", 2, True]]}],
            }
        ]
    }
    opts = normalize_run_model(run)[0]["options"]
    assert opts == [{"label": "one", "value": "1"}, {"label": "two", "value": "2"}]


# ---------------------------------------------------------------------------
# Library datasets are a single dataset, not a collection
# ---------------------------------------------------------------------------
#
# invoke_workflow's docstring advertises 'ldda' and 'ld', and Galaxy's
# run_request converts both to an HDA before the invocation starts, so a data
# slot has to take them. The rule used to be "anything that is not hda is a
# collection", which refused them and mislabelled them on the way out.


@pytest.mark.parametrize("src", ["hda", "ldda", "ld"])
def test_validate_accepts_every_single_dataset_source(src):
    res = validate_inputs(SLOTS, {"2": {"src": src, "id": LIB_ID}}, MAP)

    assert res["rejects"] == []


@pytest.mark.parametrize("src", ["ldda", "ld"])
def test_validate_does_not_call_a_library_dataset_a_collection(src):
    """The old message said 'got a collection (ldda)', which was wrong twice over."""
    res = validate_inputs(SLOTS, {"2": {"src": src, "id": LIB_ID}}, MAP)

    assert res["rejects"] == []
    everything = [r["reason"] for r in res["rejects"]] + [w["message"] for w in res["warnings"]]
    assert not any("collection" in text.lower() for text in everything)


def test_validate_still_rejects_a_collection_in_a_data_slot():
    res = validate_inputs(SLOTS, {"2": {"src": "hdca", "id": "c1"}}, MAP)

    assert len(res["rejects"]) == 1
    reason = res["rejects"][0]["reason"]
    assert "expects a single dataset" in reason
    assert "a dataset collection (hdca)" in reason


def test_the_collection_reject_does_not_pretend_galaxy_would_refuse_it():
    """Galaxy maps a workflow over a collection handed to a data input. This
    refuses it because it cannot tell that from the wrong reference, and the
    message has to say which of the two it is."""
    res = validate_inputs(SLOTS, {"2": {"src": "hdca", "id": "c1"}}, MAP)

    reason = res["rejects"][0]["reason"]
    assert "map the workflow over its elements" in reason
    assert "Unknown workflow input source" not in reason


def test_validate_reject_names_what_was_actually_passed():
    """Not just 'wrong src' -- the message has to say what arrived."""
    res = validate_inputs(SLOTS, {"0": {"src": "hdca", "id": "c1"}}, MAP)

    assert res["rejects"][0]["reason"].startswith(
        "Slot expects a single dataset (src: hda, ldda, ld); got a dataset collection (hdca)."
    )


def test_validate_reject_names_the_sources_it_accepts():
    """A closed list is only usable if the refusal says what the list is."""
    res = validate_inputs(SLOTS, {"2": {"src": "future_src", "id": "x"}}, MAP)

    assert "(src: hda, ldda, ld);" in res["rejects"][0]["reason"]


def test_validate_rejects_an_unrecognised_source():
    """Passing one through is not free: with no history_id Galaxy creates and
    commits a history before it decides the src means nothing to it."""
    res = validate_inputs(SLOTS, {"2": {"src": "future_src", "id": "x"}}, MAP)

    assert len(res["rejects"]) == 1
    assert "future_src" in res["rejects"][0]["reason"]
    assert not any("future_src" in w["message"] for w in res["warnings"])


@pytest.mark.parametrize("src", ["ldda", "ld"])
def test_a_library_reference_may_not_carry_an_ext(src):
    """Galaxy's model for one has src and id and nothing else, so an ext fails
    there whatever its value -- and fails after the history exists, which is the
    whole reason to catch it here."""
    res = validate_inputs(SLOTS, {"0": {"src": src, "id": LIB_ID, "ext": "tabular"}}, MAP)

    assert len(res["rejects"]) == 1
    reason = res["rejects"][0]["reason"]
    assert f"A library dataset reference ({src}) is just src and id" in reason
    assert "the extra key 'ext'" in reason


def test_the_library_reference_reject_names_every_stray_key():
    res = validate_inputs(
        SLOTS,
        {"0": {"src": "ldda", "id": LIB_ID, "ext": "tabular", "collection_type": "list"}},
        MAP,
    )

    assert "the extra keys 'collection_type', 'ext'" in res["rejects"][0]["reason"]


@pytest.mark.parametrize(
    ("key", "value"),
    [("map_over_type", "x"), ("hid", 3), ("workflow_step_id", "s1"), ("label", "in")],
)
def test_a_library_reference_may_carry_the_keys_bioblend_sends(key, value):
    """The model still tolerates those four, so refusing them would be a false
    reject on anything that went through bioblend."""
    res = validate_inputs(SLOTS, {"0": {"src": "ldda", "id": LIB_ID, key: value}}, MAP)

    assert res["rejects"] == []


@pytest.mark.parametrize("src", ["ldda", "ld"])
def test_a_library_reference_needs_an_id(src):
    """Nothing here ever looked at id, so a reference without one sailed through
    to Galaxy, which has it as a required field."""
    res = validate_inputs(SLOTS, {"0": {"src": src}}, MAP)

    assert len(res["rejects"]) == 1
    assert res["rejects"][0]["reason"] == (f"A library dataset reference ({src}) needs an id.")


@pytest.mark.parametrize("bad", [123, 1.5, True, None, ["d1"]])
def test_a_library_reference_id_must_be_a_string(bad):
    """Galaxy takes the id as a StrictStr, which coerces nothing."""
    res = validate_inputs(SLOTS, {"0": {"src": "ldda", "id": bad}}, MAP)

    assert len(res["rejects"]) == 1
    assert "The id on a library dataset reference must be a string" in (res["rejects"][0]["reason"])


def test_a_library_reference_id_must_not_be_empty():
    """The model would take it, but decode_id will not, and that is just as far
    past the point where the history exists."""
    res = validate_inputs(SLOTS, {"0": {"src": "ldda", "id": ""}}, MAP)

    assert len(res["rejects"]) == 1
    assert res["rejects"][0]["reason"] == (
        "The id on a library dataset reference must not be empty."
    )


@pytest.mark.parametrize("hid", [None, 3, 0, -1, True, False, 10**30])
def test_a_library_reference_hid_may_be_any_whole_number(hid):
    """A bool is an int in Python and the model takes one too, and an int of any
    size passes there, so none of these may be refused."""
    res = validate_inputs(SLOTS, {"0": {"src": "ldda", "id": LIB_ID, "hid": hid}}, MAP)

    assert res["rejects"] == []


@pytest.mark.parametrize("hid", ["abc", "", 3.5, ["3"], {"a": 1}, "1e2", 1e19])
def test_a_library_reference_hid_must_be_a_whole_number(hid):
    res = validate_inputs(SLOTS, {"0": {"src": "ldda", "id": LIB_ID, "hid": hid}}, MAP)

    assert len(res["rejects"]) == 1
    assert (
        "The hid on a library dataset reference must be a whole number or null"
        in (res["rejects"][0]["reason"])
    )


@pytest.mark.parametrize("hid", ["3", "3.0", 3.0, "3.0000000000000001", "9" * 400])
def test_a_library_reference_hid_is_refused_where_galaxy_would_coerce(hid):
    """Deliberately stricter than Galaxy. Its field is not strict, so some of
    these coerce there -- but reproducing that grammar for a legacy key is not
    worth it, and getting it subtly wrong is how "1e2" and a 400-digit string
    ended up on the opposite side of the line from where Galaxy puts them. A
    refusal costs a retry; a miss costs a history."""
    res = validate_inputs(SLOTS, {"0": {"src": "ldda", "id": LIB_ID, "hid": hid}}, MAP)

    assert len(res["rejects"]) == 1
    assert "send it as a number" in res["rejects"][0]["reason"]


@pytest.mark.parametrize("key", ["map_over_type", "workflow_step_id", "label"])
@pytest.mark.parametrize("bad", [3, True, 3.5, ["x"], {"a": 1}])
def test_the_library_reference_legacy_string_fields_must_be_strings(key, bad):
    res = validate_inputs(SLOTS, {"0": {"src": "ldda", "id": LIB_ID, key: bad}}, MAP)

    assert len(res["rejects"]) == 1
    assert (
        f"The {key} on a library dataset reference must be a string or null"
        in (res["rejects"][0]["reason"])
    )


@pytest.mark.parametrize("key", ["map_over_type", "workflow_step_id", "label"])
def test_the_library_reference_legacy_string_fields_may_be_null(key):
    res = validate_inputs(SLOTS, {"0": {"src": "ldda", "id": LIB_ID, key: None}}, MAP)

    assert res["rejects"] == []


@pytest.mark.parametrize(
    "bad", ["xyz", "1", "aa", "f2db41e1fa331b3", "f2db41e1fa331b3e0", "zzzzzzzzzzzzzzzz"]
)
def test_a_library_reference_id_must_look_like_an_encoded_id(bad):
    """decode_id hex-decodes the id and hands it to the cipher, so a non-hex
    string, an odd-length one and one that is not a whole number of blocks each
    come back as a MalformedId -- after the history exists."""
    res = validate_inputs(SLOTS, {"0": {"src": "ldda", "id": bad}}, MAP)

    assert len(res["rejects"]) == 1
    assert "does not look like a Galaxy encoded id" in res["rejects"][0]["reason"]


@pytest.mark.parametrize("ok", ["f2db41e1fa331b3e", "F2DB41E1FA331B3E", "f2db41e1fa331b3e" * 2])
def test_a_library_reference_id_may_be_any_whole_number_of_blocks(ok):
    """encode_id pads to whole 8-byte blocks, so 16 hex digits or a multiple of
    them, and the hex codec takes either case."""
    res = validate_inputs(SLOTS, {"0": {"src": "ldda", "id": ok}}, MAP)

    assert res["rejects"] == []


def test_a_library_reference_reports_every_problem_it_has():
    """One reject per thing wrong, so a caller fixes them in one pass rather than
    one round trip each."""
    res = validate_inputs(SLOTS, {"0": {"src": "ldda", "hid": "abc", "ext": "bam"}}, MAP)

    reasons = sorted(r["reason"] for r in res["rejects"])
    assert len(reasons) == 3
    assert "forbids the extra key 'ext'" in reasons[0]
    assert reasons[1].endswith("needs an id.")
    assert "must be a whole number or null" in reasons[2]


@pytest.mark.parametrize("src", ["ldda", "ld"])
def test_a_bare_library_reference_still_draws_the_unknown_datatype_warning(src):
    """Accepting the src must not skip the checks that follow it. Nothing resolves
    an ext for a library dataset, so the slot's datatype stays unproven."""
    res = validate_inputs(SLOTS, {"0": {"src": src, "id": LIB_ID}}, MAP)

    assert res["rejects"] == []
    assert any("Could not determine datatype" in w["message"] for w in res["warnings"])


@pytest.mark.parametrize(
    "ref",
    [
        {"src": "hda", "id": "d1", "ext": "bed"},
        {"src": "hda"},
        {"src": "hda", "id": 7},
        {"src": "hda", "id": ""},
    ],
)
def test_the_library_checks_do_not_touch_an_hda_reference(ref):
    """Galaxy refuses all four, and this accepts all four, exactly as it did
    before any of this. An hda's ext is where the datatype check has always got
    its value and the server writes that key itself, so there is no telling the
    server's value from the caller's by the time this sees the reference; the
    rest goes with it. Left alone, not endorsed."""
    res = validate_inputs(SLOTS, {"0": ref}, MAP)

    assert res["rejects"] == []


@pytest.mark.parametrize("src", ["hda", "ldda", "ld"])
def test_a_dataset_source_in_a_collection_slot_is_still_rejected(src):
    """The collection slot's rule is unchanged."""
    res = validate_inputs(SLOTS, {"1": {"src": src, "id": LIB_ID}}, MAP)

    assert any("expects a dataset collection" in r["reason"] for r in res["rejects"])


@pytest.mark.parametrize("src", [["hda"], {"a": 1}, 7, None])
def test_validate_does_not_crash_on_a_non_string_src(src):
    """A dict lookup on an arbitrary JSON value raises; the old != comparison did not.

    Worse than a crash: invoke_workflow's preflight catches everything and falls
    back to an empty verdict, so one bad value would disable every other check in
    the same call.
    """
    res = validate_inputs(SLOTS, {"2": {"src": src, "id": "x"}}, MAP)

    assert len(res["rejects"]) == 1
    assert "expects a single dataset" in res["rejects"][0]["reason"]


def test_one_bad_value_does_not_disable_the_other_checks():
    supplied = {
        "2": {"src": ["hda"], "id": "x"},
        "1": {"src": "hda", "id": "d1"},
    }

    res = validate_inputs(SLOTS, supplied, MAP)

    steps = {r["step_index"] for r in res["rejects"]}
    assert steps == {1, 2}  # the collection slot is still judged


def test_a_collection_element_is_rejected_from_a_data_slot():
    """Galaxy's request model parses a dce, but run_request has no branch for it,
    so it is the one refusal here that Galaxy would also make. Call it an element
    rather than a collection, since a dce may wrap either."""
    res = validate_inputs(SLOTS, {"2": {"src": "dce", "id": "e1"}}, MAP)

    assert len(res["rejects"]) == 1
    assert "a collection element (dce)" in res["rejects"][0]["reason"]


# ---------------------------------------------------------------------------
# A url reference is Galaxy's, but not this checker's
# ---------------------------------------------------------------------------
#
# run_request does dereference one into a new HDA. Its request model is strict
# about the shape, though -- the location under url or location, a datatype under
# ext, filetype or extension with the last of those winning, and nothing else
# allowed -- and a preflight that reads only some of that is worse than one that
# says up front it does not read it.


def test_a_url_reference_is_refused_for_now():
    res = validate_inputs(
        SLOTS, {"2": {"src": "url", "url": "https://example/x.txt", "ext": "txt"}}, MAP
    )

    assert len(res["rejects"]) == 1
    reason = res["rejects"][0]["reason"]
    assert "a url reference (url)" in reason
    assert "Galaxy does take one on a data input" in reason
    assert "pass the hda" in reason


@pytest.mark.parametrize("ext", [["bam"], {"a": 1}, 7])
def test_a_non_string_ext_does_not_crash_the_validator(ext):
    """The datatype lookup is a dict lookup, so a list ext raised straight out of
    validate_inputs, and invoke_workflow's blanket handler would have turned that
    into an empty verdict that disabled every other check in the call. An hda is
    the one src that can still get here with one: the server resolves the ext
    itself and leaves the caller's value alone when that lookup fails."""
    res = validate_inputs(SLOTS, {"0": {"src": "hda", "id": "d1", "ext": ext}}, MAP)

    assert res["rejects"] == []
    assert any("Could not determine datatype" in w["message"] for w in res["warnings"])
