"""Galaxy MCP - Model Context Protocol server for Galaxy bioinformatics platform.

The server's names are resolved on first use rather than bound when this package is
imported, so that importing galaxy_mcp does not build the server -- see ``__getattr__``
below. They all still answer, ``from galaxy_mcp import *`` included, but they answer at
runtime: a type checker reading ``from galaxy_mcp import *`` in a consumer will not see
them, so import them by name or from ``galaxy_mcp.server`` where a checker has to follow
them. ``__dir__`` lists the same names, which is also why ``help(galaxy_mcp)`` now
documents the server's names along with this package's own.
"""

# Imported as a module rather than ``from importlib import import_module`` because
# whatever this file leaves in its own namespace ends up in ``__all__``, and
# ``importlib`` is a name the old star import already carried out of the server.
import importlib
from typing import Any

__version__ = "1.11.0.dev0"
__author__ = "Dannon Baker"
__email__ = "dannon.baker@gmail.com"

# Input helpers that used to live in this package directly. The modules moved under ops;
# the import paths stayed, and reaching for one of them has no business building a server.
_MOVED_TO_OPS = ("tool_inputs", "workflow_inputs")

# Root names that ``from .server import *`` used to carry because the server imported them,
# and no longer does. They answer from the layer they moved to, which costs no server.
_OPS_ALIASES = {"is_reference": "galaxy_mcp.ops.tool_inputs"}


def __getattr__(name: str) -> Any:
    """Resolve on first use what ``from .server import *`` used to bind at import time.

    ``__main__`` sets the discovery mode and only then imports the server, because the
    FastMCP instance is built at module load. Importing it from here defeats that ordering,
    but nobody reaching for ``galaxy_mcp.mcp`` should have to know it: the name still works,
    and the server is imported when something actually asks for it.
    """
    if name == "__all__":
        # The one lookup that has to build the server, and the only one that should. A star
        # import asks for everything by definition, so there is nothing left to defer: PEP
        # 562 routes ``__all__`` through here when ``from galaxy_mcp import *`` looks for
        # it, and what this answers is exactly what that binds. Without it the star import
        # falls back to the module dict and carries none of the names below.
        return [n for n in __dir__() if not n.startswith("_")]
    if name.startswith("_"):
        # The star import never bound private names, and refusing them here keeps a stray
        # dunder probe from dragging the server in behind everyone's back.
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in _MOVED_TO_OPS:
        return importlib.import_module(f".{name}", __name__)
    if name in _OPS_ALIASES:
        return getattr(importlib.import_module(_OPS_ALIASES[name]), name)
    server = importlib.import_module(".server", __name__)
    try:
        return getattr(server, name)
    except AttributeError:
        pass
    # Importing the server binds galaxy_mcp.auth and the other submodules on this package.
    if name in globals():
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    server = importlib.import_module(".server", __name__)
    return sorted(
        set(globals())
        | set(_MOVED_TO_OPS)
        | set(_OPS_ALIASES)
        | {n for n in dir(server) if not n.startswith("_")}
    )
