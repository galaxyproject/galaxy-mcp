"""History commands for Galaxy CLI."""

from typing import Annotated

import typer

from galaxy_mcp.server import (
    create_history,
    get_histories,
    get_history_contents,
    get_history_details,
    list_history_ids,
)

from ..output import output_error, output_result

app = typer.Typer(name="history", help="Manage Galaxy histories", no_args_is_help=True)


def _fn(tool):
    """Extract the underlying function from a FastMCP tool."""
    return tool.fn if hasattr(tool, "fn") else tool


@app.command("list")
def list_histories(
    limit: Annotated[int | None, typer.Option("-l", "--limit", help="Maximum histories")] = None,
    offset: Annotated[int, typer.Option("-o", "--offset", help="Skip this many")] = 0,
    name: Annotated[str | None, typer.Option("-n", "--name", help="Filter by name")] = None,
) -> None:
    """List user's histories with optional pagination."""
    try:
        result = _fn(get_histories)(limit=limit, offset=offset, name=name)
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)


@app.command("ids")
def ids() -> None:
    """Get a simplified list of history IDs and names."""
    try:
        result = _fn(list_history_ids)()
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)


@app.command("create")
def create(
    name: Annotated[str, typer.Argument(help="Name for the new history")],
) -> None:
    """Create a new history."""
    try:
        result = _fn(create_history)(history_name=name)
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)


@app.command("details")
def details(
    history_id: Annotated[str, typer.Argument(help="History ID")],
) -> None:
    """Get metadata and summary for a history."""
    try:
        result = _fn(get_history_details)(history_id=history_id)
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)


@app.command("contents")
def contents(
    history_id: Annotated[str, typer.Argument(help="History ID")],
    limit: Annotated[int, typer.Option("-l", "--limit", help="Maximum items")] = 100,
    offset: Annotated[int, typer.Option("-o", "--offset", help="Skip this many")] = 0,
    deleted: Annotated[bool, typer.Option("--deleted", help="Include deleted")] = False,
    hidden: Annotated[bool, typer.Option("--hidden", help="Include hidden")] = False,
    order: Annotated[
        str, typer.Option("--order", help="Sort order (hid-asc, hid-dsc, create_time-dsc, etc.)")
    ] = "hid-asc",
) -> None:
    """Get paginated contents of a history."""
    try:
        # visible is the inverse of hidden
        result = _fn(get_history_contents)(
            history_id=history_id,
            limit=limit,
            offset=offset,
            deleted=deleted,
            visible=not hidden,
            order=order,
        )
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)
