"""The checked-in MCP surface manifest has to match the tools the server registers.

`tests/testdata/mcp-surface.json` is what the TypeScript parity check compares its
ops against, so a tool added, renamed, or re-signatured here has to land in the
manifest in the same commit -- otherwise the cross-language check goes on passing
against a description of a server that no longer exists.
"""

import asyncio
import json
from itertools import zip_longest

import pytest
from fastmcp.tools import Tool

from galaxy_mcp import server
from tests.surface_manifest import (
    CONDITIONAL_TOOLS,
    MANIFEST_PATH,
    REGENERATE_COMMAND,
    build_manifest,
    render,
)

_STALE = f"{MANIFEST_PATH.name} is stale -- regenerate it with `{REGENERATE_COMMAND}`."


@pytest.fixture(scope="module")
def generated() -> dict:
    return build_manifest()


@pytest.fixture(scope="module")
def checked_in() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def test_manifest_lists_exactly_the_registered_tools(generated, checked_in):
    have = [t["name"] for t in checked_in["tools"]]
    want = [t["name"] for t in generated["tools"]]
    assert have == want, f"{_STALE} Added: {sorted(set(want) - set(have))}. " + (
        f"Removed: {sorted(set(have) - set(want))}."
    )


def test_manifest_describes_each_tool_the_same_way(generated, checked_in):
    have = {t["name"]: t for t in checked_in["tools"]}
    for tool in generated["tools"]:
        assert have.get(tool["name"]) == tool, f"{tool['name']} changed. {_STALE}"


def test_manifest_file_is_byte_identical_to_the_generator(generated):
    want = render(generated).splitlines()
    have = MANIFEST_PATH.read_text().splitlines()
    for lineno, (w, h) in enumerate(zip_longest(want, have), start=1):
        assert h == w, f"{MANIFEST_PATH.name} differs from the generator at line {lineno}. {_STALE}"


def test_manifest_covers_conditionally_registered_tools(checked_in):
    """Tools behind an optional extra are in the manifest even when it isn't installed."""
    by_name = {t["name"]: t for t in checked_in["tools"]}
    for name, spec in CONDITIONAL_TOOLS.items():
        assert name in by_name, f"{name} is missing from the manifest. {_STALE}"
        assert by_name[name]["conditionalOn"] == spec["extra"]


@pytest.mark.parametrize("name", sorted(CONDITIONAL_TOOLS))
def test_conditional_tool_declaration_matches_the_live_registration(name):
    """The generator restates the tags of tools it may have to synthesize; catch drift."""
    spec = CONDITIONAL_TOOLS[name]
    registered = {t.name: t for t in asyncio.run(server.mcp.list_tools(run_middleware=False))}
    if name not in registered:
        pytest.skip(f"{name} needs the '{spec['extra']}' extra")
    synthesized = Tool.from_function(getattr(server, name), tags=set(spec["tags"]))
    assert synthesized.tags == registered[name].tags
    assert synthesized.parameters == registered[name].parameters
