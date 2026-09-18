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

MANIFEST_PATH = Path(__file__).parent / "testdata" / "mcp-surface.json"
REGENERATE_COMMAND = "uv run python -m tests.surface_manifest"

# Tools FastMCP registers only when an optional extra is installed. Synthesizing
# them from the plain module-level function keeps the manifest identical on a
# machine that has the extra and one that does not, so the surface contract does
# not depend on how the generating environment was provisioned.
CONDITIONAL_TOOLS: dict[str, dict[str, Any]] = {
    "recommend_biocontainer": {
        "extra": "container-recommend",
        "tags": ["extended", "read", "tools"],
    },
}


def _entry(tool: Tool, conditional_on: str | None) -> dict[str, Any]:
    entry: dict[str, Any] = {"name": tool.name, "tags": sorted(tool.tags)}
    if conditional_on:
        entry["conditionalOn"] = conditional_on
    entry["inputSchema"] = tool.parameters
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

    registered = {t.name: t for t in asyncio.run(server.mcp.list_tools(run_middleware=False))}
    entries = [
        _entry(tool, CONDITIONAL_TOOLS.get(name, {}).get("extra"))
        for name, tool in registered.items()
    ]
    for name, spec in CONDITIONAL_TOOLS.items():
        if name in registered:
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
