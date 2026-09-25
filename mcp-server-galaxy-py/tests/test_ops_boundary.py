"""galaxy_mcp.ops is the layer that owes nothing to the caller it serves.

Read statically rather than by importing: what matters is what the module declares, and a
runtime check would pass simply because something else had already pulled bioblend in. The
rule is narrow on purpose -- the standard library, and each other. A helper that needs a
client, a session or the toolbox is not this layer, whatever else it is.
"""

import ast
import pathlib
import sys

OPS = pathlib.Path(__file__).resolve().parents[1] / "src" / "galaxy_mcp" / "ops"


def _imported_roots(path):
    """Every top-level package this module reaches for, relative imports excluded."""
    roots = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _modules():
    return sorted(OPS.glob("*.py"))


def test_there_is_something_to_check():
    assert [p.name for p in _modules()] != ["__init__.py"], "no ops modules found"


def test_ops_imports_only_the_standard_library():
    outside = {}
    for path in _modules():
        beyond = _imported_roots(path) - set(sys.stdlib_module_names) - {"galaxy_mcp"}
        if beyond:
            outside[path.name] = sorted(beyond)
    assert not outside, (
        f"galaxy_mcp.ops reaches outside the standard library: {outside}. "
        "Logic that needs a third-party package belongs outside the pure operation layer."
    )


def _is_ops(module):
    return module == "galaxy_mcp.ops" or module.startswith("galaxy_mcp.ops.")


def test_ops_does_not_reach_back_out_of_the_layer():
    """The dependency runs one way: a caller imports ops, never the reverse."""
    reaching = {}
    for path in _modules():
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom) or node.level:
                continue
            module = node.module or ""
            if module.startswith("galaxy_mcp") and not _is_ops(module):
                reaching.setdefault(path.name, []).append(module)
    assert not reaching, (
        f"ops imports from outside the layer: {reaching}. "
        "The operation logic is what callers depend on, not the other way round."
    )
