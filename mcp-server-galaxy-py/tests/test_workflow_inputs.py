import json as _json
from pathlib import Path

from galaxy_mcp.workflow_inputs import normalize_ga_steps, normalize_run_model, subtype_satisfies

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
