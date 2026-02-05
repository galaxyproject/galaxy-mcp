"""IWC (Intergalactic Workflow Commission) commands for Galaxy CLI."""

from typing import Annotated

import typer

from galaxy_mcp.server import (
    get_iwc_workflow_details,
    get_iwc_workflows,
    import_workflow_from_iwc,
    recommend_iwc_workflows,
    search_iwc_workflows,
)

from ..output import output_error, output_result

app = typer.Typer(name="iwc", help="Browse and import IWC workflows", no_args_is_help=True)


def _fn(tool):
    """Extract the underlying function from a FastMCP tool."""
    return tool.fn if hasattr(tool, "fn") else tool


@app.command("list")
def list_workflows() -> None:
    """List all workflows in the IWC manifest."""
    try:
        result = _fn(get_iwc_workflows)()
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)


@app.command("search")
def search(
    query: Annotated[str, typer.Argument(help="Search query")],
) -> None:
    """Search IWC workflows by name, description, or tags."""
    try:
        result = _fn(search_iwc_workflows)(query=query)
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)


@app.command("details")
def details(
    trs_id: Annotated[str, typer.Argument(help="TRS ID of the workflow")],
) -> None:
    """Get comprehensive details about an IWC workflow."""
    try:
        result = _fn(get_iwc_workflow_details)(trs_id=trs_id)
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)


@app.command("recommend")
def recommend(
    intent: Annotated[str, typer.Argument(help="Natural language description of analysis goal")],
    limit: Annotated[int, typer.Option("-l", "--limit", help="Maximum recommendations")] = 5,
) -> None:
    """
    Get workflow recommendations based on natural language intent.

    Uses BM25 ranking to find workflows matching your analysis description.

    Examples:
        gxy iwc recommend "differential expression from RNA-seq"
        gxy iwc recommend "assemble bacterial genome from nanopore reads" --limit 3
    """
    try:
        result = _fn(recommend_iwc_workflows)(intent=intent, limit=limit)
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)


@app.command("import")
def import_wf(
    trs_id: Annotated[str, typer.Argument(help="TRS ID of the workflow to import")],
) -> None:
    """Import an IWC workflow into your Galaxy instance."""
    try:
        result = _fn(import_workflow_from_iwc)(trs_id=trs_id)
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)
