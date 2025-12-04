"""Tests for OAuth-aware functionality."""

from unittest.mock import patch

import pytest
from galaxy_mcp.auth import GalaxyCredentials
from galaxy_mcp.server import _OAuthPublicRoutes, ensure_connected
from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.testclient import TestClient


@pytest.mark.usefixtures("_test_env")
def test_ensure_connected_prefers_oauth_session(mock_galaxy_instance):
    """ensure_connected should build a Galaxy client from the active OAuth session."""
    credentials = GalaxyCredentials(
        galaxy_url="https://oauth.galaxy/",
        api_key="oauth-api-key",
        username="oauth-user",
        user_email="oauth@example.com",
        expires_at=1_700_000_000,
        scopes=["galaxy:full"],
        client_id="client-123",
    )

    with patch("galaxy_mcp.server.auth_provider", object()):
        with patch(
            "galaxy_mcp.server.get_active_session",
            return_value=(credentials, credentials.api_key),
        ):
            with patch(
                "galaxy_mcp.server.GalaxyInstance", return_value=mock_galaxy_instance
            ) as mock_constructor:
                state = ensure_connected()

    assert state["source"] == "oauth"
    assert state["connected"] is True
    assert state["gi"] is mock_galaxy_instance
    assert state["url"] == credentials.galaxy_url
    mock_constructor.assert_called_once_with(url=credentials.galaxy_url, key=credentials.api_key)


class _DummyProvider:
    def __init__(self):
        self.login_calls = 0
        self.metadata_calls = 0

    def get_login_paths(self, base_path=None):
        return {"/galaxy-auth/login"}

    def get_resource_metadata_paths(self, base_path=None):
        return {"/.well-known/oauth-protected-resource"}

    async def handle_login(self, request):
        self.login_calls += 1
        return PlainTextResponse("login-ok")

    async def handle_resource_metadata(self, request):
        self.metadata_calls += 1
        return JSONResponse({"resource": "https://oauth.galaxy/"})


def test_oauth_public_routes_expose_login_and_metadata():
    """_OAuthPublicRoutes should short-circuit auth for public endpoints."""

    app = Starlette()

    async def ping(_request):  # pragma: no cover - trivial handler
        return PlainTextResponse("pong")

    app.add_route("/ping", ping, methods=["GET"])

    provider = _DummyProvider()
    wrapped = _OAuthPublicRoutes(app, provider, base_path=None)

    # The wrapper should proxy common Starlette attributes
    assert wrapped.state is app.state
    assert wrapped.router is app.router

    client = TestClient(wrapped)

    # Public endpoints are handled by the provider
    login_response = client.get("/galaxy-auth/login")
    assert login_response.status_code == 200
    assert login_response.text == "login-ok"
    assert provider.login_calls == 1

    metadata_response = client.get("/.well-known/oauth-protected-resource")
    assert metadata_response.status_code == 200
    assert metadata_response.json() == {"resource": "https://oauth.galaxy/"}
    assert provider.metadata_calls == 1

    # Non-public endpoints fall back to the underlying application
    ping_response = client.get("/ping")
    assert ping_response.status_code == 200
    assert ping_response.text == "pong"

    # Unsupported methods should flow through to Starlette and return 404
    fallback_response = client.post("/.well-known/oauth-protected-resource")
    assert fallback_response.status_code == 404
