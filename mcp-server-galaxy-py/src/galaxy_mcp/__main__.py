"""Command-line entry point for Galaxy MCP server."""

from .server import run_http_server


def run() -> None:
    """Run the MCP server using HTTP-based transport."""
    run_http_server()


if __name__ == "__main__":
    run()
