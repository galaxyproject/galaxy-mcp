"""Server commands for Galaxy CLI."""

import typer

from galaxy_mcp.server import get_server_info

from ..output import output_error, output_result

app = typer.Typer(name="server", help="Galaxy server information", no_args_is_help=True)


def _fn(tool):
    """Extract the underlying function from a FastMCP tool."""
    return tool.fn if hasattr(tool, "fn") else tool


@app.command("info")
def info() -> None:
    """Get Galaxy server information including version and configuration."""
    try:
        result = _fn(get_server_info)()
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)
