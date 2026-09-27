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

An import written as a call is read the way the interpreter reads it. ``import_module``
resolves a relative name against its ``package`` argument when it is given one, positionally
or by keyword, so that argument is the anchor here as well -- ``import_module(".server",
package="galaxy_mcp")`` is ``galaxy_mcp.server`` and not the layer's own sibling.
``__import__`` takes its anchor from the calling module and its depth from ``level``, so that
is what is used for it. The functions are found by what each file binds them to rather than
by their usual spelling: ``from importlib import import_module as load`` and ``import
importlib as il`` and ``load = importlib.import_module`` all make a call an import, and a
name that is never bound to them is no escape hatch either.

An import whose target is computed rather than written down cannot be resolved here at all,
and is refused for that reason: the layer is meant to be checkable by reading it. That
covers a name built at runtime, an anchor built at runtime, and an importer that arrived by
a route the file does not spell out -- ``getattr(importlib, "import_module")``, an importlib
handed in as an argument, an import function pulled out of a dictionary. The checker refuses
what it cannot resolve rather than guessing what it probably means.

The directory and the package are arguments rather than constants, which is what lets a
test point the check at a module it wrote itself.
"""

import ast
import dataclasses
import pathlib
import sys

OPS_PACKAGE = "galaxy_mcp.ops"
OPS_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "galaxy_mcp" / "ops"

# The third-party packages the layer may reach for. Empty on purpose -- the input contracts
# need json, re and typing -- and an entry here is a decision about what the layer is.
ALLOWED_THIRD_PARTY: frozenset[str] = frozenset()

ALLOWED_ROOTS = frozenset(sys.stdlib_module_names) | ALLOWED_THIRD_PARTY

# The module that hands out import functions, and the two functions themselves. A file may
# call them under any name it likes, so what each file binds them to is collected first and
# its calls are judged against that rather than against these spellings.
IMPORTLIB = "importlib"
IMPORT_MODULE = "import_module"
BUILTIN_IMPORT = "__import__"
IMPORT_FUNCTIONS = frozenset({IMPORT_MODULE, BUILTIN_IMPORT})

COMPUTED_IMPORT = "dynamic import with a computed name"


@dataclasses.dataclass
class _Importers:
    """What a single file can import through.

    ``modules`` are the names bound to importlib itself, ``functions`` maps each name bound
    to an import function to which function it is, and ``opaque`` holds the names that came
    out of importlib by a route this file does not spell out -- calling through one of those
    is an import nobody can resolve by reading the source.
    """

    modules: set[str]
    functions: dict[str, str]
    opaque: set[str]


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


def _assigned_names(node: ast.Assign | ast.AnnAssign) -> list[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return [target.id for target in targets if isinstance(target, ast.Name)]


def _mentions_importer(value: ast.expr, importers: _Importers) -> bool:
    """Whether an expression has an import function or an importlib anywhere inside it."""
    watched = (
        importers.modules | set(importers.functions) | importers.opaque | {IMPORTLIB}
    ) | IMPORT_FUNCTIONS
    for node in ast.walk(value):
        if isinstance(node, ast.Name) and node.id in watched:
            return True
        if isinstance(node, ast.Attribute) and node.attr in IMPORT_FUNCTIONS:
            return True
    return False


def _binding(value: ast.expr, importers: _Importers) -> tuple[str, str] | None:
    """What an assignment's right-hand side makes of its target, if anything."""
    if isinstance(value, ast.Name):
        if value.id in importers.opaque:
            return ("opaque", "")
        if value.id in importers.functions:
            return ("function", importers.functions[value.id])
        if value.id in importers.modules:
            return ("module", "")
    if (
        isinstance(value, ast.Attribute)
        and value.attr in IMPORT_FUNCTIONS
        and isinstance(value.value, ast.Name)
        and value.value.id in importers.modules
    ):
        return ("function", value.attr)
    if _mentions_importer(value, importers):
        return ("opaque", "")
    return None


def importers_in(tree: ast.Module) -> _Importers:
    """Every name a file can import through, however it came by it.

    ``__import__`` is there from the start because it is a builtin; the rest are whatever the
    file's own imports and assignments bind. Assignments are re-read until nothing new turns
    up, so a name that is passed along a chain of them is still recognised.
    """
    importers = _Importers(set(), {BUILTIN_IMPORT: BUILTIN_IMPORT}, set())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == IMPORTLIB:
                    importers.modules.add(alias.asname or IMPORTLIB)
                elif alias.name.startswith(f"{IMPORTLIB}.") and not alias.asname:
                    # ``import importlib.util`` binds the root package as well.
                    importers.modules.add(IMPORTLIB)
        elif isinstance(node, ast.ImportFrom) and node.module == IMPORTLIB and not node.level:
            for alias in node.names:
                if alias.name in IMPORT_FUNCTIONS:
                    importers.functions[alias.asname or alias.name] = alias.name

    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign | ast.AnnAssign)]
    settled = False
    while not settled:
        settled = True
        for node in assignments:
            binding = _binding(node.value, importers) if node.value is not None else None
            if binding is None:
                continue
            kind, function = binding
            for name in _assigned_names(node):
                if kind == "module":
                    added = name not in importers.modules
                    importers.modules.add(name)
                elif kind == "function" and importers.functions.get(name, function) == function:
                    added = name not in importers.functions
                    importers.functions[name] = function
                else:
                    # Either plainly unreadable, or one name standing for two different
                    # import functions, which is no more readable than the first case.
                    added = name not in importers.opaque
                    importers.opaque.add(name)
                settled = settled and not added
    return importers


def _call_function(func: ast.expr, importers: _Importers) -> str | None:
    """Which import function a call goes through, or the refusal if that cannot be read."""
    if isinstance(func, ast.Name):
        if func.id in importers.opaque:
            return COMPUTED_IMPORT
        if func.id in importers.functions:
            return importers.functions[func.id]
        # A bare ``import_module`` the file never bound came from somewhere unreadable.
        return COMPUTED_IMPORT if func.id in IMPORT_FUNCTIONS else None
    if isinstance(func, ast.Attribute) and func.attr in IMPORT_FUNCTIONS:
        if isinstance(func.value, ast.Name) and func.value.id in importers.modules:
            return func.attr
        # ``something.import_module(...)``: an import through an importlib that reached this
        # call by a route the file does not spell out.
        return COMPUTED_IMPORT
    return None


def _argument(call: ast.Call, position: int, keyword: str) -> ast.expr | None:
    if len(call.args) > position:
        return call.args[position]
    for word in call.keywords:
        if word.arg == keyword:
            return word.value
    return None


def _string(value: ast.expr | None) -> str | None:
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return None


def _call_import(call: ast.Call, package: str, importers: _Importers) -> list[str]:
    """The module name a call imports, if it imports one at all."""
    function = _call_function(call.func, importers)
    if function is None:
        return []
    if function == COMPUTED_IMPORT:
        return [COMPUTED_IMPORT]
    name = _string(_argument(call, 0, "name"))
    if name is None:
        return [COMPUTED_IMPORT]
    if function == IMPORT_MODULE:
        anchor = _argument(call, 1, "package")
        if anchor is not None:
            # An anchor of its own replaces the file's package, exactly as import_module
            # does it. One that is computed leaves the name unresolvable.
            resolved_anchor = _string(anchor)
            if resolved_anchor is None:
                return [COMPUTED_IMPORT]
            package = resolved_anchor
        return [_resolve(name, package)]
    level = _argument(call, 4, "level")
    if level is not None:
        if not isinstance(level, ast.Constant) or not isinstance(level.value, int):
            return [COMPUTED_IMPORT]
        name = "." * level.value + name
    return [_resolve(name, package)]


def _imported_names(node: ast.AST, package: str, importers: _Importers) -> list[str]:
    """The module names one statement or call imports, relative ones resolved."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.Call):
        return _call_import(node, package, importers)
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
        tree = ast.parse(path.read_text())
        importers = importers_in(tree)
        names = {
            name
            for node in ast.walk(tree)
            for name in _imported_names(node, own_package, importers)
            if not _is_allowed(name, package)
        }
        if names:
            refused[path.relative_to(directory).as_posix()] = sorted(names)
    return refused
