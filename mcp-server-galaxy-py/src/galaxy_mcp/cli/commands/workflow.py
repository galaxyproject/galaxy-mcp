"""Workflow commands for Galaxy CLI."""

import json
from typing import Annotated, Any

import typer

from galaxy_mcp.server import (
    cancel_workflow_invocation,
    get_invocations,
    get_workflow_details,
    invoke_workflow,
    list_workflows,
)

from ..output import output_error, output_result

app = typer.Typer(name="workflow", help="Manage and run workflows", no_args_is_help=True)


def _fn(tool):
    """Extract the underlying function from a FastMCP tool."""
    return tool.fn if hasattr(tool, "fn") else tool


@app.command("list")
def list_wf(
    workflow_id: Annotated[str | None, typer.Option("--id", help="Specific workflow ID")] = None,
    name: Annotated[str | None, typer.Option("-n", "--name", help="Filter by name")] = None,
    published: Annotated[bool, typer.Option("--published", help="Include published")] = False,
) -> None:
    """List workflows available in Galaxy."""
    try:
        result = _fn(list_workflows)(workflow_id=workflow_id, name=name, published=published)
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)


@app.command("details")
def details(
    workflow_id: Annotated[str, typer.Argument(help="Workflow ID")],
    version: Annotated[int | None, typer.Option("-v", "--version", help="Specific version")] = None,
) -> None:
    """Get detailed information about a workflow."""
    try:
        result = _fn(get_workflow_details)(workflow_id=workflow_id, version=version)
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)


@app.command("invoke")
def invoke(
    workflow_id: Annotated[str, typer.Argument(help="Workflow ID")],
    inputs: Annotated[
        str | None, typer.Option("-i", "--inputs", help="Workflow inputs as JSON")
    ] = None,
    params: Annotated[
        str | None, typer.Option("-p", "--params", help="Tool parameters as JSON")
    ] = None,
    history_id: Annotated[
        str | None, typer.Option("-h", "--history-id", help="Target history ID")
    ] = None,
    history_name: Annotated[
        str | None, typer.Option("--history-name", help="Name for new history")
    ] = None,
    inputs_by: Annotated[
        str,
        typer.Option("--inputs-by", help="How to identify inputs (step_index, step_uuid, name)"),
    ] = "step_index",
) -> None:
    """
    Invoke (run) a workflow.

    INPUTS and PARAMS should be JSON objects.

    Example:
        gxy workflow invoke abc123 -i '{"0": {"src": "hda", "id": "def456"}}'
    """
    inputs_dict: dict[str, Any] | None = None
    params_dict: dict[str, Any] | None = None

    if inputs:
        try:
            inputs_dict = json.loads(inputs)
        except json.JSONDecodeError as e:
            output_error(f"Invalid JSON for inputs: {e}")
            raise typer.Exit(1)

    if params:
        try:
            params_dict = json.loads(params)
        except json.JSONDecodeError as e:
            output_error(f"Invalid JSON for params: {e}")
            raise typer.Exit(1)

    try:
        result = _fn(invoke_workflow)(
            workflow_id=workflow_id,
            inputs=inputs_dict,
            params=params_dict,
            history_id=history_id,
            history_name=history_name,
            inputs_by=inputs_by,
        )
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)


@app.command("invocations")
def invocations(
    invocation_id: Annotated[
        str | None, typer.Option("--id", help="Specific invocation ID")
    ] = None,
    workflow_id: Annotated[
        str | None, typer.Option("-w", "--workflow-id", help="Filter by workflow")
    ] = None,
    history_id: Annotated[
        str | None, typer.Option("-h", "--history-id", help="Filter by history")
    ] = None,
    limit: Annotated[int | None, typer.Option("-l", "--limit", help="Maximum to return")] = None,
    detailed: Annotated[bool, typer.Option("--detailed", help="Include step details")] = False,
) -> None:
    """View workflow invocations."""
    try:
        view = "element" if detailed else "collection"
        result = _fn(get_invocations)(
            invocation_id=invocation_id,
            workflow_id=workflow_id,
            history_id=history_id,
            limit=limit,
            view=view,
            step_details=detailed,
        )
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)


@app.command("cancel")
def cancel(
    invocation_id: Annotated[str, typer.Argument(help="Invocation ID to cancel")],
) -> None:
    """Cancel a running workflow invocation."""
    try:
        result = _fn(cancel_workflow_invocation)(invocation_id=invocation_id)
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)
