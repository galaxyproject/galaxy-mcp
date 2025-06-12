"""Command-line entry point for Galaxy MCP server."""

from . import server


def run():
    """Run the MCP server."""
    # Use FastMCP's simplified run method
    server.mcp.run()


if __name__ == "__main__":
    run()
