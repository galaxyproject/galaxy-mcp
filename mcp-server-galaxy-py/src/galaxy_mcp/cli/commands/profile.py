"""Profile commands for Galaxy CLI."""

import typer

from galaxy_mcp.server import GalaxyResult

from ..config import list_profiles
from ..output import output_error, output_result

app = typer.Typer(
    name="profile", help="Inspect configured connection profiles", no_args_is_help=True
)


@app.command("list")
def list_cmd() -> None:
    """List connection profiles from the gxy config file and planemo. No connection needed."""
    try:
        names = list_profiles()
        output_result(
            GalaxyResult(
                data=names,
                success=True,
                message=f"{len(names)} profiles available",
                count=len(names),
            )
        )
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)
