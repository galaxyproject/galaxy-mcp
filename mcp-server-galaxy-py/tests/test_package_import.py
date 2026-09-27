"""Importing the package must not build the server.

``__main__`` sets ``GALAXY_MCP_DISCOVERY_MODE`` and only then imports ``server``, because the
FastMCP instance is constructed at module load and the CodeMode transform is applied there. An
eager import in ``__init__`` silently defeats that ordering, leaving ``--discovery-mode`` with
nothing left to affect. Each check runs in a fresh subprocess: module-load state is what is being
asserted, and a sibling test that has already imported the server would decide the answer.

Not building it is only half the story. The package used to export everything ``from .server
import *`` bound, and the two input helper modules used to sit directly under ``galaxy_mcp``.
Both are names a caller outside this repository may already import, so they are checked here
too: the root names resolve on first use, and the old module paths still answer.
"""

import importlib
import os
import subprocess
import sys
import textwrap

import pytest


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


def test_importing_the_package_does_not_import_fastmcp():
    """The server is what reaches for FastMCP, so it is the tell that the server came along."""
    assert (
        _run(
            """
        import sys

        import galaxy_mcp  # noqa: F401

        print("fastmcp" in sys.modules)
        """
        )
        == "False"
    )


def test_the_server_is_still_reachable_from_the_package_root():
    """``from galaxy_mcp import mcp`` is what callers wrote before the eager import went away."""
    assert (
        _run(
            """
        from galaxy_mcp import mcp
        from galaxy_mcp.server import mcp as direct

        print(mcp is direct)
        """
        )
        == "True"
    )


def test_a_root_name_is_what_imports_the_server():
    """Nothing is imported until a name is asked for, and asking is enough to get it."""
    assert (
        _run(
            """
        import sys

        import galaxy_mcp

        before = "galaxy_mcp.server" in sys.modules
        galaxy_mcp.galaxy_state
        print(not before and "galaxy_mcp.server" in sys.modules)
        """
        )
        == "True"
    )


def test_the_root_lists_the_names_it_resolves_lazily():
    import galaxy_mcp

    listed = dir(galaxy_mcp)
    expected = {
        "mcp",
        "galaxy_state",
        "tool_inputs",
        "workflow_inputs",
        "is_reference",
        "__version__",
    }
    assert expected <= set(listed)
    assert listed == sorted(listed)


def test_a_name_the_server_stopped_importing_still_answers_at_the_root():
    """is_reference was a root name on main because server imported it; now ops answers."""
    assert (
        _run(
            """
        import sys

        from galaxy_mcp import is_reference
        from galaxy_mcp.ops.tool_inputs import is_reference as moved

        print(
            is_reference is moved
            and "galaxy_mcp.server" not in sys.modules
            and "fastmcp" not in sys.modules
        )
        """
        )
        == "True"
    )


@pytest.mark.parametrize("name", ["tool_inputs", "workflow_inputs"])
def test_the_moved_input_helpers_keep_their_old_import_path(name):
    """The modules moved under ops; the paths they moved from still answer with the same names."""
    old = importlib.import_module(f"galaxy_mcp.{name}")
    new = importlib.import_module(f"galaxy_mcp.ops.{name}")
    exported = [n for n in dir(new) if not (n.startswith("__") and n.endswith("__"))]
    assert exported, f"galaxy_mcp.ops.{name} exports nothing to compare"
    assert [n for n in dir(old) if not (n.startswith("__") and n.endswith("__"))] == exported
    assert all(getattr(old, n) is getattr(new, n) for n in exported)


def test_reaching_for_a_moved_helper_does_not_build_the_server():
    """The point of the move was a layer with no server in it; the old path keeps that."""
    assert (
        _run(
            """
        import sys

        import galaxy_mcp.tool_inputs  # noqa: F401
        import galaxy_mcp.workflow_inputs  # noqa: F401

        print("galaxy_mcp.server" in sys.modules or "fastmcp" in sys.modules)
        """
        )
        == "False"
    )
