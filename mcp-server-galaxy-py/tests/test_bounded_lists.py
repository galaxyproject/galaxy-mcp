"""Pagination behaviour of the list-returning tools.

Every tool here used to return its whole result set, which MCP clients truncate.
Each one is checked for the same six things: the default page, an explicit
limit/offset window, the last page, an empty result, rejected parameters, and
pagination metadata that matches what was actually returned.
"""

import asyncio
import json
from unittest.mock import Mock, patch

from galaxy_mcp import server as server_module
from galaxy_mcp.server import OUTPUT_BUDGET_BYTES, galaxy_state


def connected(mock_galaxy_instance):
    return patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance})


def assert_page_is_truthful(result, *, total, returned, limit, offset):
    """The whole point of the change: the metadata has to describe the payload."""
    assert result.count == returned
    pagination = result.pagination
    assert pagination is not None
    assert pagination.total_items == total
    assert pagination.returned_items == returned
    assert pagination.limit == limit
    assert pagination.offset == offset
    assert pagination.has_next is (offset + returned < total)
    assert pagination.has_previous is (offset > 0)
    if pagination.has_next:
        assert pagination.next_offset == offset + returned
    else:
        assert pagination.next_offset is None


def walk_pages(call, *, limit, key):
    """Walk every page the way an agent would, following pagination.next_offset."""
    seen = []
    offset = 0
    while True:
        result = call(limit=limit, offset=offset)
        rows = result.data["tools"] if isinstance(result.data, dict) else result.data
        seen.extend(row[key] for row in rows)
        if not result.pagination.has_next:
            return seen
        assert result.pagination.next_offset > offset
        offset = result.pagination.next_offset


# The premise of every change in this file is a byte budget: MCP clients truncate
# tool output (one common adapter at 50 KB) and hand the model unparseable JSON.
# The budget and the measurement both come from the server, so a test cannot check a
# number the tools do not enforce or measure the payload a different way than they do.

# Items far past anything a cap could allow for. A Galaxy name, a tool command and a
# workflow readme have no maximum, so the fixtures below are what "somebody's real
# server" looks like at the wrong end of the distribution: kilobyte commands, names in
# a script that costs three bytes a character, readmes that run to pages.
CJK_NAME = "\u30b2\u30ce\u30e0\u89e3\u6790\u30d1\u30a4\u30d7\u30e9\u30a4\u30f3" * 3


def huge_user_tool(i):
    tool = json.loads(json.dumps(FAT_USER_TOOL))
    tool["id"] = f"{i:016x}"
    tool["name"] = f"{CJK_NAME} {i}"
    tool["representation"]["shell_command"] = "multiqc --no-data-dir " + "-x arg " * 340
    tool["representation"]["help"] = "\u8a73\u7d30\u306a\u8aac\u660e\u3002" * 80
    return tool


def huge_tool_entry(i):
    entry = json.loads(json.dumps(FAT_TOOL_ENTRY))
    entry["id"] = f"{FAT_TOOL_ENTRY['id']}/{i}"
    entry["name"] = f"{CJK_NAME} {i}"
    entry["description"] = "\u3053\u306e\u30c4\u30fc\u30eb\u306e\u8aac\u660e\u3002" * 40
    return entry


def huge_workflow_entry(i):
    entry = json.loads(json.dumps(FAT_WORKFLOW_ENTRY))
    entry["id"] = f"{i:016x}"
    entry["name"] = f"{CJK_NAME} {i}"
    entry["annotations"] = ["\u6ce8\u91c8\u3002" * 100]
    return entry


def huge_iwc_manifest(count):
    manifest = fat_iwc_manifest(count)
    for i, workflow in enumerate(manifest[0]["workflows"]):
        workflow["readme"] = "\u8a73\u7d30\u306a\u4f7f\u7528\u6cd5\u3002" * 400
        workflow["definition"]["name"] = f"{CJK_NAME} {i}"
        workflow["definition"]["annotation"] = (
            "\u3053\u306e\u30ef\u30fc\u30af\u30d5\u30ed\u30fc\u3002" * 60
        )
    return manifest


def huge_history(i):
    return {"id": f"{i:016x}", "name": f"{CJK_NAME} {i}"}


FAT_TOOL_ENTRY = {
    "model_class": "Tool",
    "id": (
        "toolshed.g2.bx.psu.edu/repos/iuc/trinity_align_and_estimate_abundance"
        "/trinity_align_and_estimate_abundance/2.15.1+galaxy1"
    ),
    "name": "Align reads and estimate abundance",
    "version": "2.15.1+galaxy1",
    "description": "on a de novo assembly of RNA-Seq data to generate a transcriptome",
    "labels": ["updated"],
    "edam_operations": ["operation_0524", "operation_3258", "operation_0310"],
    "edam_topics": ["topic_0622", "topic_3308"],
    "hidden": "",
    "is_workflow_compatible": True,
    "xrefs": [{"value": "biotools:trinity", "reftype": "bio.tools"}],
    "config_file": (
        "/cvmfs/main.galaxyproject.org/shed_tools/toolshed.g2.bx.psu.edu/repos/iuc"
        "/trinity_align_and_estimate_abundance/9e5cd2b2a2dd"
        "/trinity_align_and_estimate_abundance/align_and_estimate_abundance.xml"
    ),
    "link": (
        "/tool_runner?tool_id=toolshed.g2.bx.psu.edu%2Frepos%2Fiuc%2Ftrinity%2F2.15.1%2Bgalaxy1"
    ),
    "min_width": -1,
    "target": "galaxy_main",
    "panel_section_id": "rna_seq",
    "panel_section_name": "RNA Seq",
    "form_style": "regular",
    "versions": ["2.15.1+galaxy1", "2.9.1+galaxy2"],
    "tool_shed_repository": {
        "name": "trinity_align_and_estimate_abundance",
        "owner": "iuc",
        "changeset_revision": "9e5cd2b2a2dd",
        "tool_shed": "toolshed.g2.bx.psu.edu",
    },
}

FAT_WORKFLOW_ENTRY = {
    "model_class": "StoredWorkflow",
    "id": "f2db41e1fa331b3e",
    "update_time": "2026-01-15T12:00:00.000000",
    "create_time": "2025-11-02T09:31:00.000000",
    "name": "RNA-seq differential expression (human, paired-end, batch-corrected)",
    "url": "/api/workflows/f2db41e1fa331b3e",
    "owner": "someuser",
    "number_of_steps": 24,
    "published": False,
    "deleted": False,
    "hidden": False,
    "tags": ["rnaseq", "deseq2", "human"],
    "latest_workflow_uuid": "6f1a2b3c-4d5e-6f70-8192-a3b4c5d6e7f8",
    "annotations": ["standard lab pipeline, see the wiki before editing"],
    "license": "MIT",
    "source_metadata": None,
    "show_in_tool_panel": False,
}

FAT_USER_TOOL = {
    "id": "b1c2d3e4f5a6b7c8",
    "uuid": "9f8e7d6c-5b4a-3928-1706-f5e4d3c2b1a0",
    "tool_id": "my_custom_qc",
    "name": "Custom QC summariser",
    "active": True,
    "representation": {
        "class": "GalaxyUserTool",
        "id": "my_custom_qc",
        "name": "Custom QC summariser",
        "version": "0.3.1",
        "container": "quay.io/biocontainers/multiqc:1.21--pyhdfd78af_0",
        "shell_command": "multiqc --no-data-dir -o out $(inputs) && mv out/*.html result.html",
        "inputs": [
            {"name": f"report_{i}", "type": "data", "format": "txt", "optional": False}
            for i in range(6)
        ],
        "outputs": [{"name": "summary", "type": "data", "format": "html"}],
        "help": "Summarises QC reports across a run. " * 6,
    },
}


def fat_iwc_manifest(count):
    """Manifest entries whose enriched summaries are as large as IWC's largest."""
    return [
        {
            "workflows": [
                {
                    "trsID": f"#workflow/github.com/iwc-workflows/long-named-workflow-{i}/main",
                    "readme": "Detailed usage notes for this workflow. " * 40,
                    "categories": ["Transcriptomics", "Variant Calling"],
                    "definition": {
                        "name": f"Comprehensive analysis workflow number {i}",
                        "annotation": "A thorough description of what this workflow does. " * 8,
                        "tags": ["rnaseq", "transcriptomics", "human", "paired-end"],
                        "license": "MIT",
                        "creator": [
                            {"name": f"Author {n}", "identifier": f"0000-0002-0000-000{n}"}
                            for n in range(4)
                        ],
                        "steps": {
                            str(s): {
                                "tool_id": (
                                    f"toolshed.g2.bx.psu.edu/repos/iuc/tool{s}/tool{s}/1.0"
                                ),
                                "type": "tool",
                            }
                            for s in range(24)
                        },
                    },
                }
                for i in range(count)
            ]
        }
    ]


def response_json(mock_galaxy_instance, payload):
    response = Mock()
    response.json.return_value = payload
    mock_galaxy_instance.make_get_request.return_value = response
    mock_galaxy_instance.url = "http://localhost:8080/api"


def client_bytes(tool_name, result):
    """What a client is handed for this result, measured through FastMCP itself.

    Deliberately not the server's own measurement. A test that asks the code under test
    how big its answer is agrees with it by construction -- swap the server's ruler for a
    shorter one and every budget test still passes. This runs the result through the
    conversion a real call goes through (FunctionTool.convert_result, which ends at the
    serializer FastMCP hands the transport) and counts the bytes of the text block, which
    is the JSON an adapter shows the model.
    """
    tool = asyncio.run(server_module.mcp.get_tool(tool_name))
    return len(tool.convert_result(result).content[0].text.encode("utf-8"))


def assert_fits_budget(tool_name, result, label):
    size = client_bytes(tool_name, result)
    assert size <= OUTPUT_BUDGET_BYTES, f"{label} reaches the client as {size:,} bytes"
    return size


def assert_the_fixture_is_over_budget(tool_name, call, limit, label):
    """The corpus has to be one the budget actually has to cut, or nothing is proved.

    An empty result fits every budget and so does a small one; either would let a test
    that only checks "the page fits" pass while the cutting never ran at all.
    """
    with patch.object(server_module, "OUTPUT_BUDGET_BYTES", 10**9):
        uncut = call(limit, 0)
    assert uncut.count > 0, f"{label}: the fixture and query produced nothing to cut"
    size = client_bytes(tool_name, uncut)
    assert size > OUTPUT_BUDGET_BYTES, (
        f"{label}: an uncut page is only {size:,} bytes, so the budget is never reached"
    )
    return uncut.count


def walk_every_page(tool_name, call, items_of, identity, label, limit):
    """Page from 0 to the end, checking each page fits and nobody is seen twice."""
    assert_the_fixture_is_over_budget(tool_name, call, limit, label)
    seen = []
    offset = 0
    for _ in range(500):
        result = call(limit, offset)
        assert_fits_budget(tool_name, result, f"{label} page at offset {offset}")
        seen.extend(identity(item) for item in items_of(result))
        if not result.pagination.has_next:
            assert len(seen) == len(set(seen)), f"{label} returned an item twice"
            return seen
        assert result.pagination.next_offset == offset + result.count
        offset = result.pagination.next_offset
    raise AssertionError(f"{label} never reached its last page")
