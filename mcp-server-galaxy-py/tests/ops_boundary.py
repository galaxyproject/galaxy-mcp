"""What galaxy_mcp.ops may import, read out of the source rather than by importing it.

A runtime check would pass simply because something else had already pulled bioblend in, so
this reads the import statements instead. The rule is an allow-list rather than a list of
banned names: the standard library, the layer's own modules, and whatever third party the
layer genuinely needs, which is nothing today. Everything else is refused, so a dependency
nobody thought to forbid fails here anyway -- the server, the auth provider, bioblend,
fastmcp, whichever HTTP client someone reaches for next year.

Both forms and both spellings count. ``import galaxy_mcp.server`` is as much a dependency as
``from galaxy_mcp.server import mcp``, and ``from ..server import mcp`` is the same import
written relatively, so a relative name is resolved against the package the file belongs to
before it is judged. The directory and the package are arguments rather than constants,
which is what lets a test point the check at a module it wrote itself.
"""

import ast
import pathlib
import sys

OPS_PACKAGE = "galaxy_mcp.ops"
OPS_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "galaxy_mcp" / "ops"

# The third-party packages the layer may reach for. Empty on purpose -- the input contracts
# need json, re and typing -- and an entry here is a decision about what the layer is.
ALLOWED_THIRD_PARTY: frozenset[str] = frozenset()

ALLOWED_ROOTS = frozenset(sys.stdlib_module_names) | ALLOWED_THIRD_PARTY


def modules_in(directory: pathlib.Path) -> list[pathlib.Path]:
    """Every module in the layer, ``__init__`` included."""
    return sorted(directory.glob("*.py"))


def _imported_names(node: ast.AST, package: str) -> list[str]:
    """The module names one statement imports, relative ones resolved against `package`."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if not isinstance(node, ast.ImportFrom):
        return []
    if not node.level:
        return [node.module or ""]
    parts = package.split(".")
    if node.level > len(parts):
        # Climbing past the top-level package. Whatever that lands on is not ours to allow,
        # and the dots are the only honest name for it.
        return ["." * node.level + (node.module or "")]
    base = ".".join(parts[: len(parts) - node.level + 1])
    if node.module:
        return [f"{base}.{node.module}"]
    # ``from . import x`` names a sibling module, not the package it sits in.
    return [f"{base}.{alias.name}" for alias in node.names]


def _is_allowed(name: str, package: str) -> bool:
    if name == package or name.startswith(f"{package}."):
        return True
    return name.split(".")[0] in ALLOWED_ROOTS


def forbidden_imports(directory: pathlib.Path, package: str = OPS_PACKAGE) -> dict[str, list[str]]:
    """Module names imported under `directory` that a module of `package` may not import."""
    refused: dict[str, list[str]] = {}
    for path in modules_in(directory):
        names = {
            name
            for node in ast.walk(ast.parse(path.read_text()))
            for name in _imported_names(node, package)
            if not _is_allowed(name, package)
        }
        if names:
            refused[path.name] = sorted(names)
    return refused
