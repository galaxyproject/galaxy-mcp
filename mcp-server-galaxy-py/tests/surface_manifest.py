"""Generate the checked-in snapshot of this server's MCP tool surface.

The snapshot is the contract the TypeScript ops in `galaxy-agent-tools` are
compared against. It is checked in rather than built on the fly so that a change
to a tool's parameters has to arrive as a reviewable diff, and so the two
languages cannot drift while a hand-maintained list of names keeps passing.

Regenerate with `uv run python -m tests.surface_manifest`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastmcp.tools import Tool

from galaxy_mcp import server
from galaxy_mcp.version import TOOL_REQUIREMENTS

MANIFEST_PATH = Path(__file__).parent / "testdata" / "mcp-surface.json"
REGENERATE_COMMAND = "uv run python -m tests.surface_manifest"

# Tools FastMCP registers only when an optional extra is present. Synthesizing
# them from the plain module-level function keeps the manifest identical on a
# machine that has the extra and one that does not, so the surface contract does
# not depend on how the generating environment was provisioned. When the extra
# is available, absence from the server is a registration bug, not a reason to
# invent a tool in the manifest.
CONDITIONAL_TOOLS: dict[str, dict[str, Any]] = {
    "recommend_biocontainer": {
        "extra": "container-recommend",
        "tags": ["extended", "read", "tools"],
        "available": lambda: server._container_recommender_available(),
    },
}


def _annotations(tool: Tool) -> dict[str, Any]:
    """The MCP annotations the tool advertises, empty when it advertises none.

    Recorded even though no tool sets one today: the read/write split lives in the
    tags, so a hint added later would change what a client is told about the tool
    while leaving every other field of this snapshot alone.
    """
    if tool.annotations is None:
        return {}
    return tool.annotations.model_dump(exclude_none=True)


# The largest integer a JSON number survives as itself. Anything past it is read
# back by the TypeScript side as a different number, so two declared defaults that
# differ would compare equal there and nobody would ever see it.
EXACT_INTEGER_LIMIT = 2**53 - 1


def _assert_numbers_survive_json(value: Any, where: str) -> None:
    if isinstance(value, dict):
        for key, each in value.items():
            _assert_numbers_survive_json(each, f"{where}.{key}")
    elif isinstance(value, list):
        for index, each in enumerate(value):
            _assert_numbers_survive_json(each, f"{where}[{index}]")
    elif isinstance(value, bool):
        return
    elif isinstance(value, int) and abs(value) > EXACT_INTEGER_LIMIT:
        raise ValueError(
            f"{where} is {value}, which no JSON reader can hand back unchanged; "
            "the manifest is the contract the TypeScript ops are compared against, "
            "so a number it cannot carry has to stay out of the surface"
        )


def _entry(tool: Tool, conditional_on: str | None) -> dict[str, Any]:
    entry: dict[str, Any] = {"name": tool.name, "tags": sorted(tool.tags)}
    # Recorded structurally rather than left to the description, so the TypeScript side's
    # `requires` can be compared against this one instead of against English.
    requires = TOOL_REQUIREMENTS.get(tool.name)
    if requires:
        entry["requires"] = {"galaxy": requires}
    if conditional_on:
        entry["conditionalOn"] = conditional_on
    entry["annotations"] = _annotations(tool)
    entry["inputSchema"] = tool.parameters
    _assert_numbers_survive_json(entry, tool.name)
    return entry


def build_manifest() -> dict[str, Any]:
    """Describe every tool the server can register, in a deterministic order."""
    # `code` discovery mode collapses the catalog into three meta-tools, which is a
    # different surface from the one this manifest describes.
    if server._discovery_mode != "full":
        raise RuntimeError(
            f"GALAXY_MCP_DISCOVERY_MODE={server._discovery_mode!r} changes the tool "
            "catalog; generate the manifest with the default 'full' mode."
        )

    # `run_middleware=False` on purpose: the tag filter is read from the environment,
    # and the snapshot describes the whole catalog rather than whatever one machine
    # happens to be showing. The filter can only hide tools, never change what one
    # takes. What it cannot do is decide between two registrations of one name --
    # over the wire FastMCP shows the highest version of each, and picking a
    # different one here would snapshot a tool nobody is served.
    listed = asyncio.run(server.mcp.list_tools(run_middleware=False))
    twice = sorted({t.name for t in listed if sum(1 for o in listed if o.name == t.name) > 1})
    if twice:
        raise ValueError(
            f"{', '.join(twice)} is registered more than once, and this snapshot has no way to "
            "say which registration a client is served; give the tools distinct names or teach "
            "the generator how the server chooses between them"
        )
    registered = {t.name: t for t in listed}
    entries = [
        _entry(tool, CONDITIONAL_TOOLS.get(name, {}).get("extra"))
        for name, tool in registered.items()
    ]
    for name, spec in CONDITIONAL_TOOLS.items():
        if name in registered:
            continue
        if spec["available"]():
            continue
        synthesized = Tool.from_function(getattr(server, name), tags=set(spec["tags"]))
        entries.append(_entry(synthesized, spec["extra"]))
    entries.sort(key=lambda e: str(e["name"]))

    return {
        "$comment": (
            "Generated snapshot of the Galaxy MCP tool surface -- do not edit by hand. "
            f"Regenerate with `{REGENERATE_COMMAND}`."
        ),
        "source": "mcp-server-galaxy-py/src/galaxy_mcp/server.py",
        "toolCount": len(entries),
        "tools": entries,
    }


def render(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2) + "\n"


def main() -> None:
    MANIFEST_PATH.write_text(render(build_manifest()))
    print(f"wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
