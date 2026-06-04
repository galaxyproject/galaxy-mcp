from galaxy_mcp.workflow_inputs import subtype_satisfies

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
