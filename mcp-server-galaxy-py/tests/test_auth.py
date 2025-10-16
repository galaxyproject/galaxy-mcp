"""Tests for Galaxy MCP authentication helpers."""

from __future__ import annotations

import json
from urllib.parse import urlparse

import pytest
from starlette.routing import Route
from starlette.testclient import TestClient

from fastmcp.server.server import FastMCP

from galaxy_mcp.auth import GalaxyOAuthProvider
from mcp.shared.auth import OAuthClientInformationFull


def _make_provider() -> GalaxyOAuthProvider:
    return GalaxyOAuthProvider(
        base_url="https://public.example.com/.well-known/mcp",
        galaxy_url="https://galaxy.example.com",
        session_secret="dummy-secret",
    )


def test_get_login_paths_includes_prefix_from_base_url() -> None:
    provider = _make_provider()

    # base path should include the path portion of the base URL
    parsed = urlparse("https://public.example.com/.well-known/mcp")
    paths = provider.get_login_paths(parsed.path)

    assert "/galaxy-auth/login" in paths
    assert "/.well-known/mcp/galaxy-auth/login" in paths


def test_login_routes_registered_on_fastmcp() -> None:
    provider = _make_provider()
    server = FastMCP("Test", auth=provider)

    app = server.http_app(path="/")
    route_paths = {route.path for route in app.routes if isinstance(route, Route)}

    assert "/galaxy-auth/login" in route_paths
    assert "/.well-known/mcp/galaxy-auth/login" in route_paths
    client = TestClient(app)

    parsed = urlparse("https://public.example.com/.well-known/mcp")
    for path in provider.get_login_paths(parsed.path):
        response = client.get(path)
        # Missing txn parameter should result in 400, but route should be reachable (not 401)
        assert response.status_code == 400


def test_resource_metadata_includes_expected_fields() -> None:
    provider = _make_provider()

    metadata = provider.get_resource_metadata()

    assert metadata["resource"] == "https://galaxy.example.com/"
    assert metadata["authorization_servers"] == ["https://public.example.com/.well-known/mcp"]
    assert metadata["scopes_supported"] == ["galaxy:full"]
    assert metadata["token_types_supported"] == ["Bearer"]


def test_resource_metadata_route_returns_payload() -> None:
    provider = _make_provider()
    server = FastMCP("Test", auth=provider)

    metadata = provider.get_resource_metadata()

    app = server.http_app(path="/")
    client = TestClient(app)

    parsed = urlparse("https://public.example.com/.well-known/mcp")
    for path in provider.get_resource_metadata_paths(parsed.path):
        response = client.get(path)
        assert response.status_code == 200
        assert response.json() == metadata


@pytest.mark.anyio
async def test_register_client_persists_to_registry(tmp_path) -> None:
    registry_path = tmp_path / "clients.json"
    provider = GalaxyOAuthProvider(
        base_url="https://public.example.com/.well-known/mcp",
        galaxy_url="https://galaxy.example.com",
        session_secret="dummy-secret",
        client_registry_path=registry_path,
    )

    client_info = OAuthClientInformationFull(
        client_id="test-client",
        client_secret="secret",
        redirect_uris=["https://example.com/callback"],
    )

    await provider.register_client(client_info)

    assert registry_path.exists()
    stored = json.loads(registry_path.read_text(encoding="utf-8"))
    assert stored[0]["client_id"] == "test-client"

    reloaded_provider = GalaxyOAuthProvider(
        base_url="https://public.example.com/.well-known/mcp",
        galaxy_url="https://galaxy.example.com",
        session_secret="dummy-secret",
        client_registry_path=registry_path,
    )

    loaded = await reloaded_provider.get_client("test-client")
    assert loaded is not None
    assert loaded.client_secret == "secret"
