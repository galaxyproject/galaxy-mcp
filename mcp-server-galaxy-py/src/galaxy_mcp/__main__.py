"""Command-line entry point for Galaxy MCP server."""

import argparse
import os


def run() -> None:
    """Run the MCP server using stdio or HTTP transport."""
    parser = argparse.ArgumentParser(description="Run the Galaxy MCP server.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        help="Transport to use (defaults to environment or stdio).",
    )
    parser.add_argument("--host", help="HTTP host to bind when using HTTP transports.")
    parser.add_argument(
        "--port",
        type=int,
        help="HTTP port to bind when using HTTP transports.",
    )
    parser.add_argument(
        "--path",
        help="Optional HTTP path when using streamable transports.",
    )
    parser.add_argument(
        "--discovery-mode",
        choices=["full", "code"],
        help=(
            "Tool discovery mode. 'full' (default) exposes every @mcp.tool registration. "
            "'code' wraps the server with FastMCP's experimental CodeMode, collapsing tools "
            "into search / get_schemas / run_galaxy_tool meta-tools."
        ),
    )
    args = parser.parse_args()

    # Discovery mode must be set before server import — the transform is applied at
    # FastMCP construction, which happens at module load time.
    if args.discovery_mode:
        os.environ["GALAXY_MCP_DISCOVERY_MODE"] = args.discovery_mode

    from . import server

    selected = (args.transport or os.environ.get("GALAXY_MCP_TRANSPORT") or "stdio").lower()
    if selected in {"streamable-http", "sse"}:
        server.run_http_server(
            host=args.host,
            port=args.port,
            transport=selected,
            path=args.path,
        )
    else:
        server.mcp.run()


if __name__ == "__main__":
    run()
