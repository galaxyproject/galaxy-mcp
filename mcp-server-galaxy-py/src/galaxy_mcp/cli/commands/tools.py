"""Tools commands for Galaxy CLI."""

import json
from typing import Annotated, Any

import typer

from galaxy_mcp.server import (
    get_tool_citations,
    get_tool_details,
    get_tool_panel,
    get_tool_run_examples,
    run_tool,
    search_tools_by_keywords,
    search_tools_by_name,
)

from ..output import output_error, output_result

app = typer.Typer(name="tools", help="Search and run Galaxy tools", no_args_is_help=True)


def _fn(tool):
    """Extract the underlying function from a FastMCP tool."""
    return tool.fn if hasattr(tool, "fn") else tool


@app.command("search")
def search(
    query: Annotated[str, typer.Argument(help="Search query for tool name/ID/description")],
) -> None:
    """Search Galaxy tools by name, ID, or description."""
    try:
        result = _fn(search_tools_by_name)(query=query)
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)


@app.command("keywords")
def keywords(
    keywords: Annotated[list[str], typer.Argument(help="Keywords to search for (space-separated)")],
) -> None:
    """Search Galaxy tools by multiple keywords."""
    try:
        result = _fn(search_tools_by_keywords)(keywords=keywords)
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)


@app.command("details")
def details(
    tool_id: Annotated[str, typer.Argument(help="Tool ID")],
    io_details: Annotated[
        bool, typer.Option("--io-details", "-i", help="Include input/output parameter details")
    ] = False,
) -> None:
    """Get detailed information about a specific tool."""
    try:
        result = _fn(get_tool_details)(tool_id=tool_id, io_details=io_details)
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)


@app.command("examples")
def examples(
    tool_id: Annotated[str, typer.Argument(help="Tool ID")],
    version: Annotated[
        str | None, typer.Option("--version", "-v", help="Tool version (use '*' for all)")
    ] = None,
) -> None:
    """Get example test definitions for a tool."""
    try:
        result = _fn(get_tool_run_examples)(tool_id=tool_id, tool_version=version)
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)


@app.command("citations")
def citations(
    tool_id: Annotated[str, typer.Argument(help="Tool ID")],
) -> None:
    """Get citation information for a tool."""
    try:
        result = _fn(get_tool_citations)(tool_id=tool_id)
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)


@app.command("panel")
def panel() -> None:
    """Get the tool panel structure (toolbox)."""
    try:
        result = _fn(get_tool_panel)()
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)


@app.command("run")
def run(
    history_id: Annotated[str, typer.Argument(help="History ID to run tool in")],
    tool_id: Annotated[str, typer.Argument(help="Tool ID to run")],
    inputs: Annotated[str, typer.Argument(help="Tool inputs as JSON string")],
) -> None:
    """
    Run a Galaxy tool with specified inputs.

    INPUTS should be a JSON object mapping input names to values.
    For dataset inputs, use: {"input_name": {"src": "hda", "id": "dataset_id"}}

    Example:
        gxy tools run abc123 fastqc '{"input_file": {"src": "hda", "id": "def456"}}'
    """
    try:
        inputs_dict: dict[str, Any] = json.loads(inputs)
    except json.JSONDecodeError as e:
        output_error(f"Invalid JSON for inputs: {e}")
        raise typer.Exit(1)

    try:
        result = _fn(run_tool)(history_id=history_id, tool_id=tool_id, inputs=inputs_dict)
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)
