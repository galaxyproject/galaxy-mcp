"""The checked-in MCP surface manifest has to match the tools the server registers.

`tests/testdata/mcp-surface.json` is what the TypeScript parity check compares its
ops against, so a tool added, renamed, or re-signatured here has to land in the
manifest in the same commit -- otherwise the cross-language check goes on passing
against a description of a server that no longer exists.
"""

import asyncio
import json
from copy import deepcopy
from itertools import zip_longest

import pytest
from fastmcp.tools import Tool
from mcp.types import ToolAnnotations

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
    want = render(generated)
    # read_bytes, not read_text: read_text normalizes line endings, which would let a
    # CRLF copy of the manifest pass as identical.
    have = MANIFEST_PATH.read_bytes().decode()
    if have == want:
        return
    for lineno, (w, h) in enumerate(zip_longest(want.splitlines(), have.splitlines()), start=1):
        assert h == w, f"{MANIFEST_PATH.name} differs from the generator at line {lineno}. {_STALE}"
    raise AssertionError(
        f"{MANIFEST_PATH.name} differs from the generator only in its line endings or its "
        f"final newline. {_STALE}"
    )


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
    if not spec["available"]():
        assert name not in registered
        return
    assert name in registered, f"{name} needs the '{spec['extra']}' registration"
    synthesized = Tool.from_function(getattr(server, name), tags=set(spec["tags"]))
    assert synthesized.tags == registered[name].tags
    assert synthesized.parameters == registered[name].parameters


def test_manifest_does_not_invent_a_conditional_tool_when_its_extra_is_available(monkeypatch):
    """A missing registration with its extra present must make the snapshot stale."""
    name = "recommend_biocontainer"
    original_list_tools = server.mcp.list_tools

    async def without_conditional_tool(*args, **kwargs):
        tools = await original_list_tools(*args, **kwargs)
        return [tool for tool in tools if tool.name != name]

    monkeypatch.setattr(server.mcp, "list_tools", without_conditional_tool)
    monkeypatch.setattr(server, "_container_recommender_available", lambda: True)

    assert name not in {tool["name"] for tool in build_manifest()["tools"]}


def test_manifest_records_the_annotations_a_tool_advertises(monkeypatch, checked_in):
    """A hint added to a tool has to move the snapshot; tags are not the whole story."""
    name = "get_histories"
    original_list_tools = server.mcp.list_tools

    async def with_a_hint(*args, **kwargs):
        tools = await original_list_tools(*args, **kwargs)
        return [
            tool.model_copy(update={"annotations": ToolAnnotations(readOnlyHint=True)})
            if tool.name == name
            else tool
            for tool in tools
        ]

    monkeypatch.setattr(server.mcp, "list_tools", with_a_hint)
    entry = next(tool for tool in build_manifest()["tools"] if tool["name"] == name)
    assert entry["annotations"] == {"readOnlyHint": True}
    assert entry != next(tool for tool in checked_in["tools"] if tool["name"] == name)


def test_manifest_refuses_a_number_json_cannot_carry(monkeypatch):
    """An integer past a double's reach reads back as a different number."""
    name = "get_histories"
    original_list_tools = server.mcp.list_tools

    async def with_a_huge_default(*args, **kwargs):
        tools = await original_list_tools(*args, **kwargs)
        for tool in tools:
            if tool.name != name:
                continue
            parameters = deepcopy(tool.parameters)
            parameters["properties"]["budget"] = {"type": "integer", "default": 2**53 + 1}
            return [
                t.model_copy(update={"parameters": parameters}) if t.name == name else t
                for t in tools
            ]
        return tools

    monkeypatch.setattr(server.mcp, "list_tools", with_a_huge_default)
    with pytest.raises(ValueError, match=rf"{name}.*budget.*default"):
        build_manifest()


def test_manifest_refuses_two_registrations_of_one_name(monkeypatch):
    """Two tools of one name: the wire shows the highest version, the snapshot must not guess."""
    name = "get_page"
    original_list_tools = server.mcp.list_tools

    async def with_two_of_them(*args, **kwargs):
        tools = await original_list_tools(*args, **kwargs)
        twin = next(tool for tool in tools if tool.name == name)
        return [*tools, twin.model_copy(update={"version": "2"})]

    monkeypatch.setattr(server.mcp, "list_tools", with_two_of_them)
    with pytest.raises(ValueError, match=name):
        build_manifest()
