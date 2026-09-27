"""galaxy_mcp.ops is the layer that owes nothing to the caller it serves.

The rule is narrow on purpose -- the standard library, and each other. A helper that needs a
client, a session or the toolbox is not this layer, whatever else it is. The check itself
lives in tests/ops_boundary.py; here is what it is pointed at, and the proof that it bites
when a module reaches back out however the import is written.
"""

from tests.ops_boundary import OPS_DIR, OPS_PACKAGE, forbidden_imports, modules_in


def test_there_is_something_to_check():
    """A check that finds no modules passes everything it is asked."""
    found = {path.name for path in modules_in(OPS_DIR)}
    assert found, f"no modules found under {OPS_DIR}"
    assert {"tool_inputs.py", "workflow_inputs.py"} <= found


def test_ops_imports_only_what_the_layer_is_allowed():
    refused = forbidden_imports(OPS_DIR)
    assert not refused, (
        f"galaxy_mcp.ops reaches for what the layer may not import: {refused}. "
        "Logic that needs the server, a client or a third-party package belongs outside "
        "the pure operation layer."
    )


def test_the_check_refuses_a_reach_back_out_of_the_layer(tmp_path):
    """The dependency runs one way: a caller imports ops, never the reverse."""
    layer = tmp_path / "ops"
    layer.mkdir()
    (layer / "__init__.py").write_text("")
    (layer / "absolute_from.py").write_text("from galaxy_mcp.server import mcp\n")
    (layer / "plain_import.py").write_text("import galaxy_mcp.server\n")
    (layer / "relative_from.py").write_text("from ..server import mcp\n")
    (layer / "relative_sibling.py").write_text("from .. import auth\n")
    (layer / "escaping.py").write_text("from ... import anything\n")
    (layer / "third_party.py").write_text("import bioblend\nfrom fastmcp import FastMCP\n")
    (layer / "unanticipated.py").write_text("def fetch():\n    import httpx\n")
    (layer / "allowed.py").write_text(
        "import json\nfrom typing import Any\n"
        "from galaxy_mcp.ops.tool_inputs import is_reference\n"
        "from . import plain_import\n"
    )

    assert forbidden_imports(layer, OPS_PACKAGE) == {
        "absolute_from.py": ["galaxy_mcp.server"],
        "plain_import.py": ["galaxy_mcp.server"],
        "relative_from.py": ["galaxy_mcp.server"],
        "relative_sibling.py": ["galaxy_mcp.auth"],
        "escaping.py": ["..."],
        "third_party.py": ["bioblend", "fastmcp"],
        "unanticipated.py": ["httpx"],
    }
