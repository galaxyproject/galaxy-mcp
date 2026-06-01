"""
Sanity checks for MCP tool tagging.
"""

import asyncio


def test_every_tool_is_tagged():
    """Every registered tool should have one domain, access, and tier tag."""
    from galaxy_mcp.server import mcp

    domains = {"connection", "histories", "datasets", "jobs", "tools", "workflows", "iwc", "user"}
    access = {"read", "write"}
    tiers = {"core", "extended", "niche"}

    tools = asyncio.run(mcp.list_tools(run_middleware=False))

    for tool in tools:
        assert tool.tags & domains, f"{tool.name} missing domain tag"
        assert tool.tags & access, f"{tool.name} missing access tag"
        assert tool.tags & tiers, f"{tool.name} missing tier tag"
