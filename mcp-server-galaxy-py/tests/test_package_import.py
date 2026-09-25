"""Importing the package must not build the server.

``__main__`` sets ``GALAXY_MCP_DISCOVERY_MODE`` and only then imports ``server``, because the
FastMCP instance is constructed at module load and the CodeMode transform is applied there. An
eager import in ``__init__`` silently defeats that ordering, leaving ``--discovery-mode`` with
nothing left to affect. Each check runs in a fresh subprocess: module-load state is what is being
asserted, and a sibling test that has already imported the server would decide the answer.
"""

import os
import subprocess
import sys
import textwrap


def _run(script: str) -> str:
    env = os.environ.copy()
    env.pop("GALAXY_MCP_DISCOVERY_MODE", None)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().splitlines()[-1]


def test_importing_the_package_does_not_import_the_server():
    assert (
        _run(
            """
        import sys

        import galaxy_mcp  # noqa: F401

        print("galaxy_mcp.server" in sys.modules)
        """
        )
        == "False"
    )


def test_the_package_still_reports_its_version():
    """The release tooling rewrites __version__ in __init__.py, and server.py reads the metadata."""
    import galaxy_mcp

    assert _run("import galaxy_mcp; print(galaxy_mcp.__version__)") == galaxy_mcp.__version__


def test_the_server_submodule_is_still_reachable_from_the_package():
    assert _run("from galaxy_mcp import server; print(bool(server.mcp))") == "True"


def test_the_discovery_mode_flag_reaches_the_server():
    """The regression: run() sets the env var, so server must not have been imported before it."""
    assert (
        _run(
            """
        import sys

        import fastmcp

        fastmcp.FastMCP.run = lambda self, **kwargs: None
        sys.argv = ["galaxy-mcp", "--discovery-mode", "code"]

        from galaxy_mcp.__main__ import run

        run()
        print(sys.modules["galaxy_mcp.server"]._discovery_mode)
        """
        )
        == "code"
    )
