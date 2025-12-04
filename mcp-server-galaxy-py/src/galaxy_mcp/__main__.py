"""Command-line entry point for Galaxy MCP server."""

import argparse
import os

from . import server


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
    args = parser.parse_args()

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
