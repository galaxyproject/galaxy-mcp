"""Dataset commands for Galaxy CLI."""

from typing import Annotated

import typer

from galaxy_mcp.server import (
    download_dataset,
    get_dataset_details,
    get_job_details,
    upload_file,
    upload_file_from_url,
)

from ..output import output_error, output_result

app = typer.Typer(name="dataset", help="Manage datasets", no_args_is_help=True)


def _fn(tool):
    """Extract the underlying function from a FastMCP tool."""
    return tool.fn if hasattr(tool, "fn") else tool


@app.command("details")
def details(
    dataset_id: Annotated[str, typer.Argument(help="Dataset ID")],
    preview: Annotated[
        bool, typer.Option("--preview/--no-preview", help="Include content preview")
    ] = True,
    preview_lines: Annotated[
        int, typer.Option("--preview-lines", help="Number of preview lines")
    ] = 10,
) -> None:
    """Get detailed information about a dataset."""
    try:
        result = _fn(get_dataset_details)(
            dataset_id=dataset_id,
            include_preview=preview,
            preview_lines=preview_lines,
        )
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)


@app.command("download")
def download(
    dataset_id: Annotated[str, typer.Argument(help="Dataset ID")],
    output: Annotated[str | None, typer.Option("-o", "--output", help="Output file path")] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Download even if not in 'ok' state")
    ] = False,
) -> None:
    """Download a dataset from Galaxy."""
    try:
        result = _fn(download_dataset)(
            dataset_id=dataset_id,
            file_path=output,
            require_ok_state=not force,
        )
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)


@app.command("upload")
def upload(
    path: Annotated[str, typer.Argument(help="Local file path to upload")],
    history_id: Annotated[
        str | None, typer.Option("-h", "--history-id", help="Target history ID")
    ] = None,
) -> None:
    """Upload a local file to Galaxy."""
    try:
        result = _fn(upload_file)(path=path, history_id=history_id)
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)


@app.command("upload-url")
def upload_url(
    url: Annotated[str, typer.Argument(help="URL of file to upload")],
    history_id: Annotated[
        str | None, typer.Option("-h", "--history-id", help="Target history ID")
    ] = None,
    file_type: Annotated[
        str, typer.Option("--type", help="File type (auto for auto-detection)")
    ] = "auto",
    dbkey: Annotated[str, typer.Option("--dbkey", help="Genome build")] = "?",
    name: Annotated[str | None, typer.Option("--name", help="Name for uploaded file")] = None,
) -> None:
    """Upload a file from URL to Galaxy."""
    try:
        result = _fn(upload_file_from_url)(
            url=url,
            history_id=history_id,
            file_type=file_type,
            dbkey=dbkey,
            file_name=name,
        )
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)


@app.command("job")
def job(
    dataset_id: Annotated[str, typer.Argument(help="Dataset ID")],
    history_id: Annotated[
        str | None, typer.Option("-h", "--history-id", help="History ID (for performance)")
    ] = None,
) -> None:
    """Get job details for a dataset."""
    try:
        result = _fn(get_job_details)(dataset_id=dataset_id, history_id=history_id)
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)
