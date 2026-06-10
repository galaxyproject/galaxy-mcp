"""Main CLI application for Galaxy MCP."""

import contextlib
from typing import Annotated

import typer

from galaxy_mcp.server import connect_global

from .commands import collection, dataset, history, iwc, profile, server, tools, user, workflow
from .config import load_profile
from .output import output_error, output_result, set_pretty_output

app = typer.Typer(
    name="gxy",
    help="Galaxy command-line interface for bioinformatics operations.",
    no_args_is_help=True,
)

# Register command groups
app.add_typer(tools.app, name="tools", help="Search and run Galaxy tools")
app.add_typer(history.app, name="history", help="Manage Galaxy histories")
app.add_typer(dataset.app, name="dataset", help="Manage datasets")
app.add_typer(collection.app, name="collection", help="Manage dataset collections")
app.add_typer(workflow.app, name="workflow", help="Manage and run workflows")
app.add_typer(iwc.app, name="iwc", help="Browse and import IWC workflows")
app.add_typer(server.app, name="server", help="Galaxy server information")
app.add_typer(user.app, name="user", help="User information")
app.add_typer(profile.app, name="profile", help="Inspect configured connection profiles")


@app.callback()
def main(
    ctx: typer.Context,
    url: Annotated[
        str | None,
        typer.Option("--url", "-u", help="Galaxy server URL", envvar="GALAXY_URL"),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", "-k", help="Galaxy API key", envvar="GALAXY_API_KEY"),
    ] = None,
    profile: Annotated[
        str | None,
        typer.Option("--profile", "-p", help="Config profile to use"),
    ] = None,
    pretty: Annotated[
        bool,
        typer.Option("--pretty", help="Pretty-print JSON output"),
    ] = False,
) -> None:
    """
    Galaxy CLI - Command line interface for Galaxy bioinformatics platform.

    Configure credentials via:
    - Environment variables: GALAXY_URL and GALAXY_API_KEY
    - Config file: ~/.galaxy-mcp/config.toml
    - Command line options: --url and --api-key
    """
    set_pretty_output(pretty)

    # Load configuration
    config = load_profile(profile)

    # Command-line options override config
    final_url = url or config.url
    final_api_key = api_key or config.api_key

    # Store in context for commands to use
    ctx.ensure_object(dict)
    ctx.obj["url"] = final_url
    ctx.obj["api_key"] = final_api_key
    ctx.obj["profile"] = profile

    # Seed a process-global connection for commands that talk to Galaxy.
    # Best-effort: IWC discovery commands need no connection, and `connect`
    # handles its own. Commands that DO need Galaxy will surface a clean
    # "Not connected" error if creds are missing or invalid.
    if final_url and final_api_key and ctx.invoked_subcommand not in (None, "connect"):
        with contextlib.suppress(Exception):
            connect_global(final_url, final_api_key)


@app.command()
def connect(ctx: typer.Context) -> None:
    """Connect to Galaxy and verify credentials."""
    url = ctx.obj.get("url")
    api_key = ctx.obj.get("api_key")

    if not url or not api_key:
        output_error(
            "Missing credentials. Set GALAXY_URL and GALAXY_API_KEY environment variables, "
            "use --url and --api-key options, or configure ~/.galaxy-mcp/config.toml"
        )
        raise typer.Exit(1)

    try:
        result = connect_global(url=url, api_key=api_key)
        output_result(result)
    except Exception as e:
        output_error(str(e))
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
