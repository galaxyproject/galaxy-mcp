"""Moved to :mod:`galaxy_mcp.ops.tool_inputs`; this path is kept so old imports keep working."""

from typing import Any

from galaxy_mcp.ops import tool_inputs as _moved
from galaxy_mcp.ops.tool_inputs import *  # noqa: F403

# The star import above carries the public names with their annotations; these two carry
# everything else the old path exposed, private helpers included, without listing it twice.


def __getattr__(name: str) -> Any:
    return getattr(_moved, name)


def __dir__() -> list[str]:
    return dir(_moved)
