"""galaxy_mcp.ops is the layer that owes nothing to the caller it serves.

The rule is narrow on purpose -- the standard library, and each other. A helper that needs a
client, a session or the toolbox is not this layer, whatever else it is.

Two witnesses say so, and they disagree in useful ways. The static one lives in
tests/ops_boundary.py: it judges the imports that are written down against the allow-list, and
refuses any mention of the dynamic-import machinery outright rather than trying to work out
what a call would load. It reads every file, including the ones nothing imports, but it can
only report what the source says. The runtime one is here: it imports each module of the layer
in a fresh interpreter and looks at what ended up in ``sys.modules``. It cannot see a reach-back
that never runs, but it sees through any spelling whatsoever, which is exactly what the static
rule stopped trying to do.

The fabricated layer below is the case file: every escape found in review, each one now refused
by mention, plus the handful of modules that must keep passing.
"""

import ast
import functools
import os
import pathlib
import subprocess
import sys

import pytest

from tests.ops_boundary import (
    ALLOWED_ROOTS,
    MODULE_TABLE,
    OPS_DIR,
    OPS_PACKAGE,
    boundary_refusals,
    modules_in,
    package_of,
    refusal,
)

# What no module of the layer may ever have loaded. The derived check below subsumes this
# list; it is written out so that a failure names the thing that went wrong.
NEVER_LOADED = frozenset(
    {
        "galaxy_mcp.server",
        "galaxy_mcp.auth",
        "galaxy_mcp.__main__",
        "fastmcp",
        "bioblend",
        "requests",
        "httpx",
        "mcp",
    }
)

# Importing a submodule imports its parents, so the chain above the layer is loaded whatever
# the module does. That is why ``galaxy_mcp`` is allowed here and refused by the static rule:
# the interpreter creates it, and the layer's source has no business naming it.
PACKAGE_CHAIN = frozenset(
    ".".join(OPS_PACKAGE.split(".")[: depth + 1]) for depth in range(len(OPS_PACKAGE.split(".")))
)


def ops_modules() -> list[str]:
    """Every importable name in the layer, the package itself included."""
    names = set()
    for path in modules_in(OPS_DIR):
        package = package_of(path, OPS_DIR, OPS_PACKAGE)
        names.add(package if path.name == "__init__.py" else f"{package}.{path.stem}")
    return sorted(names)


def loaded_after(statement: str, extra_path: pathlib.Path | None = None) -> set[str]:
    """What a fresh interpreter has in sys.modules once it has run `statement`."""
    env = os.environ.copy()
    env.pop("GALAXY_MCP_DISCOVERY_MODE", None)
    if extra_path is not None:
        env["PYTHONPATH"] = os.pathsep.join(
            [str(extra_path), *filter(None, [env.get("PYTHONPATH")])]
        )
    script = f"{statement}\nimport sys\nprint(repr(sorted(sys.modules)))"
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True
    )
    return set(ast.literal_eval(result.stdout.strip().splitlines()[-1]))


@functools.cache
def _interpreter_baseline() -> frozenset[str]:
    """What the interpreter loads before anything of ours runs -- site hooks and all."""
    return frozenset(loaded_after("pass"))


def outside_the_layer(loaded: set[str]) -> set[str]:
    """The modules in `loaded` that the layer is not allowed to have pulled in.

    The allowed set is the static rule's own: the layer's subtree, and the roots the
    allow-list names. Whatever a bare interpreter already had is not this layer's doing.
    """
    return {
        name
        for name in loaded - _interpreter_baseline()
        if name not in PACKAGE_CHAIN
        and not name.startswith(f"{OPS_PACKAGE}.")
        and name.split(".")[0] not in ALLOWED_ROOTS
    }


def test_there_is_something_to_check():
    """A check that finds no modules passes everything it is asked."""
    found = {path.name for path in modules_in(OPS_DIR)}
    assert found, f"no modules found under {OPS_DIR}"
    assert {"tool_inputs.py", "workflow_inputs.py"} <= found
    assert OPS_PACKAGE in ops_modules()


def test_ops_imports_only_what_the_layer_is_allowed():
    refused = boundary_refusals(OPS_DIR)
    assert not refused, (
        f"galaxy_mcp.ops reaches for what the layer may not import: {refused}. "
        "Logic that needs the server, a client or a third-party package belongs outside "
        "the pure operation layer."
    )


@pytest.mark.parametrize("module", ops_modules())
def test_importing_an_ops_module_pulls_in_nothing_outside_the_layer(module):
    """The second witness: whatever the source says, this is what the import actually loads."""
    loaded = loaded_after(f"import {module}")
    assert module in loaded
    assert not loaded & NEVER_LOADED, f"{module} loaded {sorted(loaded & NEVER_LOADED)}"
    assert not outside_the_layer(loaded), (
        f"importing {module} loaded {sorted(outside_the_layer(loaded))}, which is outside "
        "the standard library and the layer itself."
    )


def test_both_witnesses_catch_an_importer_that_arrives_as_a_default_argument(tmp_path):
    """The form three rounds of resolving dynamic imports kept missing, caught twice over.

    The static rule never works out that ``importer`` is ``__import__`` -- it refuses the file
    for saying ``__import__`` at all -- and the runtime rule does not care how it was spelled,
    only that ``galaxy_mcp.server`` ended up loaded.
    """
    (tmp_path / "ops_escape.py").write_text(
        "def load(importer=__import__):\n"
        "    return importer('galaxy_mcp.server')\n\n\n"
        "server = load()\n"
    )

    assert boundary_refusals(tmp_path, OPS_PACKAGE) == {"ops_escape.py": [refusal("__import__", 1)]}

    loaded = loaded_after("import ops_escape", tmp_path)
    assert "galaxy_mcp.server" in loaded & NEVER_LOADED
    assert "galaxy_mcp.server" in outside_the_layer(loaded)


def test_the_static_witness_alone_catches_a_loader_that_only_runs_when_called(tmp_path):
    """A stdlib loader deferred into a function loads nothing at import time, so the runtime
    witness cannot see it; refusing the loader's name is what catches it."""
    (tmp_path / "ops_deferred.py").write_text(
        "from pkgutil import resolve_name\n\n\n"
        "def load():\n"
        "    return resolve_name('galaxy_mcp.server:mcp')\n"
    )

    assert boundary_refusals(tmp_path, OPS_PACKAGE) == {
        "ops_deferred.py": [
            refusal("pkgutil", 1),
            refusal("resolve_name", 1),
            refusal("resolve_name", 5),
        ]
    }
    loaded = loaded_after("import ops_deferred", tmp_path)
    assert not (loaded & NEVER_LOADED)


def test_the_import_allow_list_is_what_stops_an_ordinary_stdlib_loader(tmp_path):
    """pydoc.locate names no importer and loads nothing until called; it is refused because
    pydoc is not something the layer may import at all."""
    (tmp_path / "ops_locate.py").write_text(
        "def load():\n    from pydoc import locate\n\n    return locate('galaxy_mcp.server.mcp')\n"
    )

    assert boundary_refusals(tmp_path, OPS_PACKAGE) == {"ops_locate.py": ["pydoc"]}
    loaded = loaded_after("import ops_locate", tmp_path)
    assert not (loaded & NEVER_LOADED)


def write_fabricated_layer(layer: pathlib.Path) -> None:
    """A layer holding every escape review has found, plus what must keep passing."""
    layer.mkdir(exist_ok=True)
    (layer / "__init__.py").write_text("")

    # Imports that are written down, judged against the allow-list.
    (layer / "absolute_from.py").write_text("from galaxy_mcp.server import mcp\n")
    (layer / "plain_import.py").write_text("import galaxy_mcp.server\n")
    (layer / "relative_from.py").write_text("from ..server import mcp\n")
    (layer / "relative_sibling.py").write_text("from .. import auth\n")
    (layer / "escaping.py").write_text("from ... import anything\n")
    (layer / "third_party.py").write_text("import bioblend\nfrom fastmcp import FastMCP\n")
    (layer / "unanticipated.py").write_text("def fetch():\n    import httpx\n")
    # Written relatively from the top of the layer, ``..tool_inputs`` is outside it; the same
    # line one package down means the layer's own module. Each file is judged by what its own
    # dots resolve to.
    (layer / "climbing_out.py").write_text("from ..tool_inputs import is_reference\n")

    # Every dynamic form below was an escape at some point, and each round closed it by
    # resolving one more binding rule. Now none of them is resolved: each file says a word it
    # may not say, and that is the whole finding.
    (layer / "dynamic_literal.py").write_text(
        "import importlib\n\nmcp = importlib.import_module('galaxy_mcp.server').mcp\n"
    )
    (layer / "dynamic_builtin.py").write_text("client = __import__('bioblend')\n")
    (layer / "dynamic_computed.py").write_text(
        "from importlib import import_module\n\n\ndef load(name):\n    return import_module(name)\n"
    )
    (layer / "package_anchor.py").write_text(
        "import importlib\n\nmcp = importlib.import_module('.server', package='galaxy_mcp').mcp\n"
    )
    (layer / "package_anchor_positional.py").write_text(
        "import importlib\n\nauth = importlib.import_module('.auth', 'galaxy_mcp')\n"
    )
    (layer / "computed_anchor.py").write_text(
        "import importlib\n\nanchor = 'galaxy' + '_mcp'\n"
        "mcp = importlib.import_module('.server', package=anchor).mcp\n"
    )
    (layer / "aliased_call.py").write_text(
        "from importlib import import_module as load\n\nserver = load('galaxy_mcp.server')\n"
    )
    (layer / "aliased_module.py").write_text(
        "import importlib as il\n\nserver = il.import_module('galaxy_mcp.server')\n"
    )
    (layer / "assigned_importer.py").write_text(
        "import importlib\n\nload = importlib.import_module\nauth = load('galaxy_mcp.auth')\n"
    )
    (layer / "rebound_builtin.py").write_text(
        "from importlib import __import__ as bring\n\nclient = bring('bioblend')\n"
    )
    (layer / "builtin_fromlist.py").write_text(
        "mcp = __import__('galaxy_mcp.server', fromlist=['mcp']).mcp\n"
    )
    (layer / "getattr_importer.py").write_text(
        "import importlib\n\nload = getattr(importlib, 'import_module')\n"
        "server = load('galaxy_mcp.server')\n"
    )
    (layer / "passed_through.py").write_text(
        "def load(importer):\n    return importer.import_module('galaxy_mcp.server')\n"
    )
    (layer / "computed_string.py").write_text(
        "from importlib import import_module\n\n"
        "server = import_module('GALAXY_MCP.SERVER'.upper().lower())\n"
    )
    (layer / "star_importlib.py").write_text(
        "from importlib import *\n\nserver = import_module('galaxy_mcp.server')\n"
    )

    # The ordinary bindings that walked past the last resolver: a default argument, a
    # staticmethod, an alias shadowed in a scope nothing calls, and the builtin fetched from
    # the module that holds it. The lambda it already refused, for the shape of the assignment
    # rather than for the name -- which is the shape this rule generalises.
    (layer / "default_arg_importer.py").write_text(
        "def load(importer=__import__):\n"
        "    return importer('galaxy_mcp.server')\n\n\n"
        "server = load()\n"
    )
    (layer / "staticmethod_importer.py").write_text(
        "from importlib import import_module\n\n\n"
        "class Loader:\n"
        "    load = staticmethod(import_module)\n\n\n"
        "server = Loader.load('galaxy_mcp.server')\n"
    )
    (layer / "shadowed_alias.py").write_text(
        "from importlib import import_module as load\n\n\n"
        "def unused():\n"
        "    from importlib import __import__ as load\n\n\n"
        "server = load('.server', package='galaxy_mcp')\n"
    )
    (layer / "from_builtins.py").write_text(
        "from builtins import __import__ as load\n\nserver = load('galaxy_mcp.server')\n"
    )
    (layer / "lambda_default.py").write_text(
        "load = lambda importer=__import__: importer('bioblend')\n\nclient = load()\n"
    )

    # Machinery reached without naming an importer: through a string, through the builtins
    # table, through source handed to the compiler, through the module table.
    (layer / "string_name.py").write_text("name = '__import__'\nload = getattr(object, name)\n")
    (layer / "builtins_table.py").write_text(
        "server = __builtins__['__import__']('galaxy_mcp.server')\n"
    )
    (layer / "exec_source.py").write_text("exec('import galaxy_mcp.server')\n")
    # The standard library's own dotted-name loaders, which name no importer at all and only
    # load when called, so the runtime witness alone would never see them.
    (layer / "pkgutil_resolve.py").write_text(
        "from pkgutil import resolve_name\n\n\n"
        "def load():\n"
        "    return resolve_name('galaxy_mcp.server:mcp')\n"
    )
    (layer / "runpy_module.py").write_text(
        "import runpy\n\n\ndef load():\n    return runpy.run_module('galaxy_mcp.server')\n"
    )
    # Ordinary standard-library modules that resolve a dotted name when called: not machinery by
    # name, refused for being outside the import allow-list.
    (layer / "pydoc_locate.py").write_text(
        "def load():\n    from pydoc import locate\n\n    return locate('galaxy_mcp.server.mcp')\n"
    )
    (layer / "pickle_restore.py").write_text(
        "import pickle\n\n\ndef restore(payload):\n    return pickle.loads(payload)\n"
    )
    (layer / "mock_patch.py").write_text(
        "from unittest import mock\n\n\n"
        "def patched():\n"
        "    return mock.patch('galaxy_mcp.server.mcp')\n"
    )
    (layer / "zip_loader.py").write_text(
        "import zipimport\n\n\n"
        "def load(archive):\n"
        "    return zipimport.zipimporter(archive).load_module('galaxy_mcp.server')\n"
    )
    (layer / "code_builtins.py").write_text(
        "def run(source):\n    return eval(compile(source, '<ops>', 'exec'), globals(), vars())\n"
    )
    (layer / "module_table.py").write_text(
        "import sys\n\nserver = sys.modules['galaxy_mcp'].server\n"
    )
    (layer / "module_table_aliased.py").write_text(
        "import sys as system\n\nsystem.modules.pop('galaxy_mcp.ops', None)\n"
    )
    (layer / "module_table_from.py").write_text(
        "from sys import modules\n\nserver = modules['galaxy_mcp'].server\n"
    )

    nested = layer / "nested"
    nested.mkdir(exist_ok=True)
    (nested / "__init__.py").write_text("")
    (nested / "helper.py").write_text("import galaxy_mcp.server\n")
    (nested / "climbing_out.py").write_text("from ...server import mcp\n")
    (nested / "allowed.py").write_text(
        "from ..tool_inputs import is_reference\nfrom . import helper\n"
    )

    # What must keep passing, including the two builtins the rule deliberately leaves alone:
    # ``re.compile`` is not the builtin ``compile``, and ``getattr`` with a name of its own is
    # how the shipped layer reads an optional attribute off an exception.
    (layer / "allowed.py").write_text(
        "import json\nimport re\nfrom typing import Any\n"
        "from galaxy_mcp.ops.tool_inputs import is_reference\n"
        "from . import plain_import\n\n"
        "DIGITS = re.compile(r'\\d+')\n\n\n"
        "def status(exc: Exception) -> Any:\n"
        "    return getattr(exc, 'status_code', None)\n"
    )


def test_the_check_refuses_a_reach_back_out_of_the_layer(tmp_path):
    """The dependency runs one way: a caller imports ops, never the reverse."""
    layer = tmp_path / "ops"
    write_fabricated_layer(layer)

    assert boundary_refusals(layer, OPS_PACKAGE) == {
        "absolute_from.py": ["galaxy_mcp.server"],
        "plain_import.py": ["galaxy_mcp.server"],
        "relative_from.py": ["galaxy_mcp.server"],
        "relative_sibling.py": ["galaxy_mcp.auth"],
        "escaping.py": ["..."],
        "third_party.py": ["bioblend", "fastmcp"],
        "unanticipated.py": ["httpx"],
        "climbing_out.py": ["galaxy_mcp.tool_inputs"],
        "dynamic_literal.py": [
            refusal("importlib", 1),
            refusal("import_module", 3),
            refusal("importlib", 3),
        ],
        "dynamic_builtin.py": [refusal("__import__", 1)],
        "dynamic_computed.py": [
            refusal("import_module", 1),
            refusal("importlib", 1),
            refusal("import_module", 5),
        ],
        "package_anchor.py": [
            refusal("importlib", 1),
            refusal("import_module", 3),
            refusal("importlib", 3),
        ],
        "package_anchor_positional.py": [
            refusal("importlib", 1),
            refusal("import_module", 3),
            refusal("importlib", 3),
        ],
        "computed_anchor.py": [
            refusal("importlib", 1),
            refusal("import_module", 4),
            refusal("importlib", 4),
        ],
        "aliased_call.py": [refusal("import_module", 1), refusal("importlib", 1)],
        "aliased_module.py": [refusal("importlib", 1), refusal("import_module", 3)],
        "assigned_importer.py": [
            refusal("importlib", 1),
            refusal("import_module", 3),
            refusal("importlib", 3),
        ],
        "rebound_builtin.py": [refusal("__import__", 1), refusal("importlib", 1)],
        "builtin_fromlist.py": [refusal("__import__", 1)],
        "getattr_importer.py": [
            refusal("importlib", 1),
            refusal("import_module", 3),
            refusal("importlib", 3),
        ],
        "passed_through.py": [refusal("import_module", 2)],
        "computed_string.py": [
            refusal("import_module", 1),
            refusal("importlib", 1),
            refusal("import_module", 3),
        ],
        "star_importlib.py": [refusal("importlib", 1), refusal("import_module", 3)],
        "default_arg_importer.py": [refusal("__import__", 1)],
        "staticmethod_importer.py": [
            refusal("import_module", 1),
            refusal("importlib", 1),
            refusal("import_module", 5),
        ],
        "shadowed_alias.py": [
            refusal("import_module", 1),
            refusal("importlib", 1),
            refusal("__import__", 5),
            refusal("importlib", 5),
        ],
        "from_builtins.py": [refusal("__import__", 1), refusal("builtins", 1)],
        "lambda_default.py": [refusal("__import__", 1)],
        "string_name.py": [refusal("__import__", 1)],
        "builtins_table.py": [refusal("__builtins__", 1), refusal("__import__", 1)],
        "exec_source.py": [refusal("exec", 1)],
        "pkgutil_resolve.py": [
            refusal("pkgutil", 1),
            refusal("resolve_name", 1),
            refusal("resolve_name", 5),
        ],
        "runpy_module.py": [refusal("runpy", 1), refusal("run_module", 5), refusal("runpy", 5)],
        "zip_loader.py": [
            refusal("zipimport", 1),
            refusal("zipimport", 5),
            refusal("zipimporter", 5),
        ],
        "pydoc_locate.py": ["pydoc"],
        "pickle_restore.py": ["pickle"],
        "mock_patch.py": ["unittest"],
        "code_builtins.py": [
            refusal("compile", 2),
            refusal("eval", 2),
            refusal("globals", 2),
            refusal("vars", 2),
        ],
        "module_table.py": ["sys", refusal(MODULE_TABLE, 3)],
        "module_table_aliased.py": ["sys", refusal(MODULE_TABLE, 3)],
        "module_table_from.py": ["sys", refusal(MODULE_TABLE, 1)],
        "nested/helper.py": ["galaxy_mcp.server"],
        "nested/climbing_out.py": ["galaxy_mcp.server"],
    }
