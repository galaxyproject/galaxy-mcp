"""Collection commands for Galaxy CLI."""

from typing import Annotated

import typer

from galaxy_mcp.server import get_collection_details

from ..output import output_error, output_result

app = typer.Typer(name="collection", help="Manage dataset collections", no_args_is_help=True)


def _fn(tool):
    """Extract the underlying function from a FastMCP tool."""
    return tool.fn if hasattr(tool, "fn") else tool


@app.command("details")
def details(
    collection_id: Annotated[str, typer.Argument(help="Collection ID")],
    max_elements: Annotated[
        int, typer.Option("--max-elements", help="Maximum elements to return")
    ] = 100,
) -> None:
    """Get detailed information about a dataset collection."""
    try:
        result = _fn(get_collection_details)(
            collection_id=collection_id,
            max_elements=max_elements,
        )
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)
