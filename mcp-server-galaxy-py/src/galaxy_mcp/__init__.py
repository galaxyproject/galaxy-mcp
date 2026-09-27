"""Galaxy MCP - Model Context Protocol server for Galaxy bioinformatics platform."""

from importlib import import_module
from typing import Any

__version__ = "1.11.0.dev0"
__author__ = "Dannon Baker"
__email__ = "dannon.baker@gmail.com"

# Input helpers that used to live in this package directly. The modules moved under ops;
# the import paths stayed, and reaching for one of them has no business building a server.
_MOVED_TO_OPS = ("tool_inputs", "workflow_inputs")


def __getattr__(name: str) -> Any:
    """Resolve on first use what ``from .server import *`` used to bind at import time.

    ``__main__`` sets the discovery mode and only then imports the server, because the
    FastMCP instance is built at module load. Importing it from here defeats that ordering,
    but nobody reaching for ``galaxy_mcp.mcp`` should have to know it: the name still works,
    and the server is imported when something actually asks for it.
    """
    if name.startswith("_"):
        # The star import never bound private names, and refusing them here keeps a stray
        # dunder probe from dragging the server in behind everyone's back.
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in _MOVED_TO_OPS:
        return import_module(f".{name}", __name__)
    server = import_module(".server", __name__)
    try:
        return getattr(server, name)
    except AttributeError:
        pass
    # Importing the server binds galaxy_mcp.auth and the other submodules on this package.
    if name in globals():
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    server = import_module(".server", __name__)
    return sorted(
        set(globals()) | set(_MOVED_TO_OPS) | {n for n in dir(server) if not n.startswith("_")}
    )
