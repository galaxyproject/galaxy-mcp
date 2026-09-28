"""What galaxy_mcp.ops may import, read out of the source rather than by importing it.

A check that ran at import time would pass simply because something else had already pulled
bioblend in, so this reads the source. Two rules, and only the first one resolves anything.

**Imports that are written down** are judged against an allow-list: a short list of pure-data
standard-library modules (``ALLOWED_IMPORTS``), the layer's own modules, and whatever third
party the layer genuinely needs, which is nothing today. ``ast.Import`` and ``ast.ImportFrom``,
absolute or relative, with a relative name resolved against the package the file itself sits
in, so the same line means different things
at different depths and is judged accordingly. Everything else is refused without being named
-- the server, the auth provider, bioblend, fastmcp, whichever HTTP client someone reaches for
next year -- which is the point of writing it as an allow-list. Nested packages are walked
too, because a reach-back would hide there otherwise.

**The dynamic-import machinery is refused by mention**, not by resolution. Earlier versions of
this file tried to work out what a call would load: which name the importer had been bound to,
which anchor the dots were measured against, which assignment had passed it along. That chase
does not end. Deciding what a name means at a call site is the whole of Python's binding
rules, so there is always one more ordinary spelling -- a default argument, a staticmethod, an
alias shadowed in another scope -- and the resolver is wrong again. The layer is pure
functions over input shapes and has nothing to load at runtime, so the rule is that naming the
machinery is itself the refusal:

* ``importlib``, ``import_module``, ``__import__``, ``builtins`` and ``__builtins__``, and the
  standard library's other loaders -- ``pkgutil`` (``resolve_name``), ``runpy`` (``run_module``,
  ``run_path``) and ``zipimport`` -- however they appear: a name, an attribute, an import
  statement, an alias, a string literal. There
  is no scoping and no resolution, so a default argument, a lambda, a class body, a decorator
  and a docstring all count. A module of this layer can be written without saying any of
  those words.
* a call to ``globals``, ``vars``, ``exec``, ``eval`` or ``compile``. These are builtins, so a
  call to one is a bare name; ``re.compile`` is a different function and is left alone.
* any ``sys.modules``. Any attribute named ``modules`` is read as ``sys.modules``, since an
  aliased ``sys`` is still ``sys`` and nothing in this layer has a ``modules`` of its own.
* a written import of anything outside ``ALLOWED_IMPORTS`` -- a short list of pure-data
  standard-library modules, not the standard library. ``pydoc.locate``, ``pickle.loads`` and
  ``unittest.mock.patch`` all turn a dotted name into a loaded module and none of them names an
  importer; rather than list every such door, the layer imports only what it is allowed to.

``getattr`` is deliberately not refused outright: ``tool_inputs`` reads an optional
``status_code`` off an exception with it, which is honest work. Reaching the machinery through
it takes the name as a string, and a machinery name written as a string is refused wherever it
appears, which is the part that matters here.

None of this proves a module loads nothing; it proves the source does not say so. A loader
object reached off something already imported -- ``re.__spec__.loader`` and its class -- names
none of these words, and no reader of the source will ever catch every such route. The second
witness is in tests/test_ops_boundary.py: it imports each module in a fresh interpreter and
looks at what ended up in ``sys.modules``, which does not care how the load was spelled.

The directory and the package are arguments rather than constants, which is what lets a test
point the check at a module it wrote itself.
"""

import ast
import pathlib
import sys

OPS_PACKAGE = "galaxy_mcp.ops"
OPS_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "galaxy_mcp" / "ops"

# The third-party packages the layer may reach for. Empty on purpose -- the input contracts
# need json, re and typing -- and an entry here is a decision about what the layer is.
ALLOWED_THIRD_PARTY: frozenset[str] = frozenset()

# What the layer's own source may import. Not "the standard library": the standard library
# holds pydoc, pickle, unittest.mock, pkgutil, runpy and other doors that turn a string into a
# loaded module, and listing doors one at a time does not converge. So the written-import rule is
# an allow-list of pure-data modules -- what the input contracts need today plus the obvious
# companions -- and a module not on it is refused for that reason alone. Adding one is a decision
# about what the layer is, made here.
ALLOWED_IMPORTS: frozenset[str] = (
    frozenset(
        {
            "abc",
            "collections",
            "copy",
            "dataclasses",
            "datetime",
            "decimal",
            "enum",
            "fractions",
            "functools",
            "itertools",
            "json",
            "math",
            "numbers",
            "operator",
            "re",
            "string",
            "textwrap",
            "types",
            "typing",
        }
    )
    | ALLOWED_THIRD_PARTY
)

# What may be found in sys.modules after importing a layer module in a clean interpreter. Wider
# than ALLOWED_IMPORTS on purpose: json, re and typing pull other standard-library modules in
# transitively, and none of those is the thing the runtime witness exists to catch.
ALLOWED_ROOTS = frozenset(sys.stdlib_module_names) | ALLOWED_THIRD_PARTY

# Every way of naming the machinery that turns a string into a loaded module.
MACHINERY = frozenset(
    {
        "importlib",
        "import_module",
        "__import__",
        "builtins",
        "__builtins__",
        # The standard library's other doors into the same machinery: pkgutil.resolve_name
        # turns "pkg.mod:attr" into an import, runpy runs a module or a path by name, and
        # zipimport loads from an archive. Naming any of them is the refusal, as above.
        "pkgutil",
        "resolve_name",
        "runpy",
        "run_module",
        "run_path",
        "zipimport",
        "zipimporter",
    }
)

# Builtins that turn a string into code or hand back a namespace to rummage through.
FORBIDDEN_CALLS = frozenset({"globals", "vars", "exec", "eval", "compile"})

# Reported for any attribute named ``modules``; the table itself is the thing to stay out of.
MODULE_TABLE = "sys.modules"

# The package root. ``import galaxy_mcp.ops.tool_inputs`` binds the name ``galaxy_mcp`` in the
# importing module, and the root resolves ``mcp`` and the tool functions lazily -- so a layer
# module that holds that name is one attribute away from waking the server, deferred past both
# witnesses. The layer imports its siblings with ``from ... import`` or relative imports and never
# holds the root's name at all.
ROOT_NAME = "galaxy_mcp"
ROOT_BINDING = "galaxy_mcp (bound by a bare import)"


def refusal(name: str, lineno: int) -> str:
    """How a mention of the machinery is reported."""
    return f"dynamic import machinery: {name} at line {lineno}"


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


def _imported_names(node: ast.AST, package: str) -> list[str]:
    """The module names one import statement names, relative ones resolved."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if not isinstance(node, ast.ImportFrom):
        return []
    if not node.level:
        return [node.module or ""]
    base = _resolve("." * node.level + (node.module or ""), package)
    if node.module or base.startswith("."):
        return [base]
    # ``from . import x`` names a sibling module, not the package it sits in.
    return [f"{base}.{alias.name}" for alias in node.names]


def _machinery_in_import(node: ast.Import | ast.ImportFrom) -> list[tuple[str, int]]:
    """The machinery an import statement names, whichever half of it says so."""
    found = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            # ``import importlib.util`` is importlib; ``import x as builtins`` is a mention
            # of the name it binds.
            for spelling in (alias.name.split(".")[0], alias.asname):
                if spelling in MACHINERY:
                    found.append((spelling, node.lineno))
        return found
    root = (node.module or "").split(".")[0]
    if root in MACHINERY:
        found.append((root, node.lineno))
    for alias in node.names:
        for spelling in (alias.name, alias.asname):
            if spelling in MACHINERY:
                found.append((spelling, node.lineno))
        if root == "sys" and alias.name == "modules":
            found.append((MODULE_TABLE, node.lineno))
    return found


def machinery_mentions(tree: ast.Module) -> list[str]:
    """Every mention of the dynamic-import machinery in a file, by line."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            found += [(line, name) for name, line in _machinery_in_import(node)]
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root == ROOT_NAME and alias.asname is None:
                        found.append((node.lineno, ROOT_BINDING))
        elif isinstance(node, ast.Name) and node.id in MACHINERY:
            found.append((node.lineno, node.id))
        elif isinstance(node, ast.Name) and node.id == ROOT_NAME:
            found.append((node.lineno, ROOT_NAME))
        elif isinstance(node, ast.Attribute):
            if node.attr in MACHINERY:
                found.append((node.lineno, node.attr))
            elif node.attr == "modules":
                found.append((node.lineno, MODULE_TABLE))
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in MACHINERY
        ):
            # A string literal, wherever it sits: a docstring is one too, and the layer can
            # describe itself without naming the machinery.
            found.append((node.lineno, node.value))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in FORBIDDEN_CALLS
        ):
            found.append((node.lineno, node.func.id))
    return [refusal(name, line) for line, name in sorted(set(found))]


def _is_allowed(name: str, package: str) -> bool:
    if name == package or name.startswith(f"{package}."):
        return True
    return name.split(".")[0] in ALLOWED_IMPORTS


def boundary_refusals(directory: pathlib.Path, package: str = OPS_PACKAGE) -> dict[str, list[str]]:
    """What the modules under `directory` do that a module of `package` may not.

    Each value is the module names the file imports and may not, followed by every mention of
    the dynamic-import machinery it makes.
    """
    refused: dict[str, list[str]] = {}
    for path in modules_in(directory):
        own_package = package_of(path, directory, package)
        tree = ast.parse(path.read_text())
        names = sorted(
            {
                name
                for node in ast.walk(tree)
                for name in _imported_names(node, own_package)
                # A machinery module is reported by the mention rule below, once, with its line.
                if not _is_allowed(name, package) and name.split(".")[0] not in MACHINERY
            }
        )
        mentions = machinery_mentions(tree)
        if names or mentions:
            refused[path.relative_to(directory).as_posix()] = names + mentions
    return refused
