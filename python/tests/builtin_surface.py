"""Generate the checked-in snapshot of GALAXY'S OWN MCP tool surface.

Galaxy ships an MCP server of its own -- `lib/galaxy/webapps/galaxy/api/mcp.py`, served
when `enable_mcp_server` is set -- which calls Galaxy's `AgentOperationsManager` in
process, where this package's tools go through bioblend and REST. It is the third
surface the parity report compares, and this is where its contract comes from.

Unlike `surface_manifest.py`, this one cannot run in this package's environment: the
tools are defined inside `get_mcp_app()` in a Galaxy module, so the only way to read what
they take is to import Galaxy. It therefore runs against a Galaxy CHECKOUT, with that
checkout's interpreter, and needs no Galaxy server, no database and no credentials -- the
app it hands `get_mcp_app` is a stub with the two attributes the setup code touches, and
FastMCP builds the schemas from the function signatures alone.

    export GALAXY_ROOT=/path/to/galaxy
    cd /path/to/galaxy-mcp
    "$GALAXY_ROOT/.venv/bin/python" -B python/tests/builtin_surface.py

Three things about that. The assignment is its own line because a `VAR=x cmd` prefix does
not reach the `$VAR` in its own arguments -- the shell expands those before it applies the
assignment, so the one-liner runs `/.venv/bin/python`. The interpreter is named rather than
discovered: it has to be one that can import Galaxy, which in a checkout set up by
`run.sh`/`make setup` is `.venv/bin/python` under the checkout, and anywhere else is
whatever environment has Galaxy installed -- nothing here goes looking, so a missing
`.venv` is a "no such file" before the generator starts. And `-B` keeps the interpreter
from writing `__pycache__` into the checkout, which importing Galaxy otherwise does; the
generator sets `sys.dont_write_bytecode` before each of its own imports for the same
reason, and the flag is the half that holds whatever else is on the way in.

CI has no Galaxy checkout, so this snapshot is NOT checked for staleness there the way
`mcp-surface.json` is. It is refreshed by hand, per Galaxy release or when somebody wants
the report to reflect newer work, and it records the commit it was taken from so a reader
always knows which Galaxy the third column describes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import Any

SNAPSHOT_PATH = Path(__file__).resolve().parents[2] / "contract" / "galaxy-builtin-surface.json"
MODULE_PATH = "lib/galaxy/webapps/galaxy/api/mcp.py"
REGENERATE_COMMAND = (
    "export GALAXY_ROOT=/path/to/galaxy && "
    '"$GALAXY_ROOT/.venv/bin/python" -B python/tests/builtin_surface.py'
)


def _read_only_imports() -> None:
    """Stop the coming imports leaving anything behind in the Galaxy checkout.

    Importing Galaxy writes `__pycache__/*.pyc` beside every module it touches, under
    whatever directory the module came from -- which is the checkout this generator only
    ever reads. Those files are git-ignored there, so a `git status` on the checkout does
    not show them, and "the checkout is untouched" would be a claim nobody could check.
    """
    sys.dont_write_bytecode = True


def _git(galaxy_root: Path, *args: str) -> str:
    """Read something out of the Galaxy checkout, without writing to it.

    "Read-only command" is not the same as "writes nothing": `git status` refreshes the
    index when the cached stat data is stale, and refreshing it means writing
    `.git/index`. On a checkout the size of Galaxy's that is likely rather than exotic --
    anything that touched the working tree since the last index write is enough.
    `--no-optional-locks` is git's own way to say "answer the question and take no locks",
    which drops the write; `core.fsmonitor=false` covers the other way a query can leave
    something behind, since a monitored repository starts a daemon and puts its socket
    under `.git`. Both are passed before the subcommand because they are top-level
    options, and both are here rather than in the calls so a call cannot forget them.
    """
    result = subprocess.run(
        ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-C", str(galaxy_root), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def provenance(galaxy_root: Path) -> dict[str, Any]:
    """Which Galaxy this snapshot describes.

    A snapshot of uncommitted code cannot be stamped with anything anyone else could
    check out, so an edited module stops the run instead. Everything else about the
    checkout may be dirty -- this reads one file's contract, not the tree's.
    """
    dirty = _git(galaxy_root, "status", "--porcelain", "--", MODULE_PATH)
    if dirty:
        raise SystemExit(
            f"{MODULE_PATH} has uncommitted changes in {galaxy_root}; this snapshot records the "
            "commit it came from, and there is no commit for what is in the working tree"
        )
    _read_only_imports()
    sys.path.insert(0, str(galaxy_root / "lib"))
    # Imported here rather than at the top: both come from the Galaxy checkout, which is only
    # on the path once the line above has run, and fastmcp is Galaxy's rather than this one's.
    import fastmcp  # noqa: PLC0415
    from galaxy.version import VERSION  # noqa: PLC0415

    return {
        "version": VERSION,
        "branch": _git(galaxy_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _git(galaxy_root, "rev-parse", "HEAD"),
        "describe": _git(galaxy_root, "describe", "--tags", "--always"),
        "commitDate": _git(galaxy_root, "log", "-1", "--format=%cI"),
        # The module's own blob, so a reader can tell whether the surface moved even when
        # the commit did: everything below is derived from this file and nothing else.
        "moduleBlob": _git(galaxy_root, "rev-parse", f"HEAD:{MODULE_PATH}"),
        # FastMCP writes the JSON Schema, so which one wrote it is part of the reading.
        "fastmcp": fastmcp.__version__,
        "capturedOn": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }


class _StubApp:
    """The least app `get_mcp_app` touches: a base URL and a model with a User class.

    Deliberately not a Galaxy app. Building one would need a config file and a database,
    and nothing below the tool signatures is read here -- an app with more on it would
    only make it harder to see that this reads a contract rather than runs a server.
    """

    class config:  # noqa: N801  (mimicking an attribute, not declaring a class)
        galaxy_infrastructure_url = "http://localhost:8080"
        # Read by newer revisions of the module when it mounts the HTTP app.
        mcp_server_path = "/api/mcp"

    class model:  # noqa: N801
        class User:
            pass


def _entry(tool: Any) -> dict[str, Any]:
    """One tool, in the same shape `mcp-surface.json` writes.

    `tags` and `annotations` are written even though the built-in server sets neither,
    because empty is what it says: every tool is registered with a bare `@mcp.tool()`, so
    a client is told nothing about which of them write. That is a difference worth being
    able to see rather than one to leave out of the file.
    """
    annotations = tool.annotations.model_dump(exclude_none=True) if tool.annotations else {}
    return {
        "name": tool.name,
        "tags": sorted(tool.tags),
        "annotations": annotations,
        "inputSchema": tool.parameters,
    }


def build_snapshot(galaxy_root: Path) -> dict[str, Any]:
    galaxy = provenance(galaxy_root)
    # Galaxy's import machinery is chatty about missing config; this reads a contract and
    # writes a file, and the noise would end up in whatever captured the output.
    logging.disable(logging.CRITICAL)
    _read_only_imports()
    from galaxy.webapps.galaxy.api.mcp import get_mcp_app  # noqa: PLC0415  (needs sys.path)

    server = get_mcp_app(_StubApp()).state.mcp_server
    listed = asyncio.run(server.list_tools(run_middleware=False))
    entries = sorted((_entry(tool) for tool in listed), key=lambda e: str(e["name"]))
    return {
        "$comment": (
            "Generated snapshot of the MCP tool surface Galaxy itself serves -- do not edit by "
            f"hand. Regenerate with `{REGENERATE_COMMAND}`. Not checked for staleness in CI: it "
            "needs a Galaxy checkout, so it is refreshed by hand and says which commit it "
            "describes."
        ),
        "source": f"galaxy:{MODULE_PATH}",
        "galaxy": galaxy,
        "toolCount": len(entries),
        "tools": entries,
    }


def render(snapshot: dict[str, Any]) -> str:
    return json.dumps(snapshot, indent=2) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--galaxy",
        default=os.environ.get("GALAXY_ROOT"),
        help="path to a Galaxy checkout (default: $GALAXY_ROOT)",
    )
    args = parser.parse_args()
    if not args.galaxy:
        raise SystemExit("say where Galaxy is: --galaxy PATH, or set GALAXY_ROOT")
    galaxy_root = Path(args.galaxy).expanduser().resolve()
    if not (galaxy_root / MODULE_PATH).is_file():
        raise SystemExit(f"{galaxy_root} does not look like a Galaxy checkout: no {MODULE_PATH}")
    SNAPSHOT_PATH.write_text(render(build_snapshot(galaxy_root)), newline="\n")
    print(f"wrote {SNAPSHOT_PATH}")


if __name__ == "__main__":
    main()
