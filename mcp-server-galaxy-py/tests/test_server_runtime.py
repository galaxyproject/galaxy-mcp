"""Tests covering runtime helpers for the MCP server."""

from __future__ import annotations

from unittest.mock import Mock

from galaxy_mcp import server


def test_run_http_server_defaults_streamable_http_to_root(monkeypatch) -> None:
    dummy_mcp = Mock()
    monkeypatch.setattr(server, "mcp", dummy_mcp)
    monkeypatch.delenv("GALAXY_MCP_HTTP_PATH", raising=False)
    monkeypatch.delenv("GALAXY_MCP_TRANSPORT", raising=False)

    server.run_http_server(host="127.0.0.1", port=9000)

    dummy_mcp.run.assert_called_once_with(
        transport="streamable-http",
        host="127.0.0.1",
        port=9000,
        path="/",
    )
