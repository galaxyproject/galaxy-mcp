"""What galaxy_mcp.ops may import, read out of the source rather than by importing it.

A runtime check would pass simply because something else had already pulled bioblend in, so
this reads the import statements instead. The rule is an allow-list rather than a list of
banned names: the standard library, the layer's own modules, and whatever third party the
layer genuinely needs, which is nothing today. Everything else is refused, so a dependency
nobody thought to forbid fails here anyway -- the server, the auth provider, bioblend,
fastmcp, whichever HTTP client someone reaches for next year.

Every spelling of an import counts, because every spelling loads the same module.
``import galaxy_mcp.server`` is as much a dependency as ``from galaxy_mcp.server import
mcp``; ``from ..server import mcp`` is the same import written relatively; and
``importlib.import_module("galaxy_mcp.server")`` is the same import written as a call. A
relative name is resolved against the package the file itself belongs to, not against the
top of the layer, so a module in a subpackage is judged by what its own dots mean. Whole
subtrees count too: a nested package is where a reach-back would hide otherwise.

An import whose target is computed rather than written down cannot be resolved here at all,
and is refused for that reason: the layer is meant to be checkable by reading it.

The directory and the package are arguments rather than constants, which is what lets a
test point the check at a module it wrote itself.
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

# The calls that import a module at runtime. ``import_module`` is matched by its own name as
# well, since it is usually reached for as ``from importlib import import_module``.
DYNAMIC_IMPORT_CALLS = frozenset({"__import__", "import_module"})

COMPUTED_IMPORT = "dynamic import with a computed name"


def modules_in(directory: pathlib.Path) -> list[pathlib.Path]:
    """Every module in the layer, ``__init__`` and nested packages included."""
    return sorted(directory.rglob("*.py"))


def package_of(path: pathlib.Path, directory: pathlib.Path, package: str) -> str:
    """The dotted package a file sits in, given that `directory` is `package`."""
    return ".".join([package, *path.relative_to(directory).parent.parts])


def _resolve(name: str, package: str) -> str:
    """A dotted name as written, with any leading dots resolved against `package`."""
    level = len(name) - len(name.lstrip("."))
    if not level:
        return name
    remainder = name[level:]
    parts = package.split(".")
    if level > len(parts):
        # Climbing past the top-level package. Whatever that lands on is not ours to allow,
        # and the dots are the only honest name for it.
        return name
    base = ".".join(parts[: len(parts) - level + 1])
    return f"{base}.{remainder}" if remainder else base


def _called_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _imported_names(node: ast.AST, package: str) -> list[str]:
    """The module names one statement or call imports, relative ones resolved."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.Call):
        if _called_name(node.func) not in DYNAMIC_IMPORT_CALLS:
            return []
        target = node.args[0] if node.args else None
        if isinstance(target, ast.Constant) and isinstance(target.value, str):
            return [_resolve(target.value, package)]
        return [COMPUTED_IMPORT]
    if not isinstance(node, ast.ImportFrom):
        return []
    if not node.level:
        return [node.module or ""]
    base = _resolve("." * node.level + (node.module or ""), package)
    if node.module or base.startswith("."):
        return [base]
    # ``from . import x`` names a sibling module, not the package it sits in.
    return [f"{base}.{alias.name}" for alias in node.names]


def _is_allowed(name: str, package: str) -> bool:
    if name == COMPUTED_IMPORT:
        return False
    if name == package or name.startswith(f"{package}."):
        return True
    return name.split(".")[0] in ALLOWED_ROOTS


def forbidden_imports(directory: pathlib.Path, package: str = OPS_PACKAGE) -> dict[str, list[str]]:
    """Module names imported under `directory` that a module of `package` may not import."""
    refused: dict[str, list[str]] = {}
    for path in modules_in(directory):
        own_package = package_of(path, directory, package)
        names = {
            name
            for node in ast.walk(ast.parse(path.read_text()))
            for name in _imported_names(node, own_package)
            if not _is_allowed(name, package)
        }
        if names:
            refused[path.relative_to(directory).as_posix()] = sorted(names)
    return refused
