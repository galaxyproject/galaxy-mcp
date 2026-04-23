"""Phase 3: CodeMode discovery mode toggle.

Verifies that setting GALAXY_MCP_DISCOVERY_MODE=code collapses the full tool
catalog into CodeMode's meta-tools (search / get_schema / run_galaxy_tool),
and that the default behavior is unchanged.

The server module instantiates FastMCP at import time, so each mode is exercised
in a fresh subprocess to avoid module-reload pollution of sibling tests.
"""

import json
import os
import subprocess
import sys
import textwrap


def _tool_names(env_mode: str | None) -> set[str]:
    env = os.environ.copy()
    env.pop("GALAXY_MCP_DISCOVERY_MODE", None)
    if env_mode is not None:
        env["GALAXY_MCP_DISCOVERY_MODE"] = env_mode

    script = textwrap.dedent(
        """
        import asyncio, json
        from galaxy_mcp import server
        tools = asyncio.run(server.mcp.list_tools(run_middleware=False))
        print(json.dumps(sorted(t.name for t in tools)))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True
    )
    return set(json.loads(result.stdout.strip().splitlines()[-1]))


def test_full_mode_exposes_all_tools():
    names = _tool_names(None)
    assert "connect" in names
    assert "get_histories" in names
    assert "run_galaxy_tool" not in names
    assert "get_schema" not in names


def test_code_mode_exposes_meta_tools_only():
    names = _tool_names("code")
    assert names == {"search", "get_schema", "run_galaxy_tool"}


def test_unknown_mode_falls_back_to_full():
    names = _tool_names("bogus")
    assert "connect" in names
    assert "run_galaxy_tool" not in names
