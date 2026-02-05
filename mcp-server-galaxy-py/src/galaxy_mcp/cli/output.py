"""Output formatting for Galaxy MCP CLI."""

import json
import sys
from typing import Any

# Global state for output formatting
_pretty_output = False


def set_pretty_output(pretty: bool) -> None:
    """Set whether to use pretty output formatting."""
    global _pretty_output
    _pretty_output = pretty


def is_pretty_output() -> bool:
    """Check if pretty output is enabled."""
    return _pretty_output


def output_result(result: Any) -> None:
    """
    Output a result to stdout.

    If the result has a model_dump method (Pydantic model), use it.
    Otherwise, output as JSON directly.

    Args:
        result: The result to output (GalaxyResult or dict)
    """
    # Handle Pydantic models
    data = result.model_dump() if hasattr(result, "model_dump") else result

    if _pretty_output:
        print(json.dumps(data, indent=2, default=str))
    else:
        print(json.dumps(data, default=str))


def output_error(message: str) -> None:
    """Output an error message to stderr."""
    error_data = {"error": message, "success": False}
    if _pretty_output:
        print(json.dumps(error_data, indent=2), file=sys.stderr)
    else:
        print(json.dumps(error_data), file=sys.stderr)
