"""Tests for Galaxy MCP authentication helpers."""

from __future__ import annotations

import json
from collections import Counter
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest
from fastmcp.server.server import FastMCP
from galaxy_mcp.auth import (
    AuthorizationTransaction,
    GalaxyAuthenticationError,
    GalaxyOAuthProvider,
)
from mcp.server.auth.provider import AuthorizationParams
from mcp.shared.auth import OAuthClientInformationFull
from starlette.routing import Route
from starlette.testclient import TestClient


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


def test_resource_metadata_endpoint_returns_expected_payload() -> None:
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

    assert metadata["resource"] == "https://galaxy.example.com/"
    assert metadata["authorization_servers"] == ["https://public.example.com/.well-known/mcp"]
    assert metadata["scopes_supported"] == ["galaxy:full"]
    assert metadata["token_types_supported"] == ["Bearer"]


@pytest.mark.asyncio()
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


def test_normalize_base_path() -> None:
    normalize = GalaxyOAuthProvider._normalize_base_path
    assert normalize(None) is None
    assert normalize("") is None
    assert normalize("/") is None
    assert normalize("prefix") == "/prefix"
    assert normalize("/prefix/") == "/prefix"


def test_get_routes_overrides_fastmcp_defaults() -> None:
    provider = _make_provider()
    server = FastMCP("Test", auth=provider)
    app = server.http_app(path="/")

    login_paths = provider.get_login_paths(urlparse(provider.base_url).path)
    metadata_paths = provider.get_resource_metadata_paths(urlparse(provider.base_url).path)

    counts = Counter(
        route.path
        for route in app.routes
        if isinstance(route, Route) and route.path in (login_paths | metadata_paths)
    )
    assert counts
    assert all(count == 1 for count in counts.values())


def test_handle_login_requires_txn_param() -> None:
    provider = _make_provider()
    server = FastMCP("Test", auth=provider)
    app = server.http_app(path="/")
    client = TestClient(app)

    response = client.get("/galaxy-auth/login")
    assert response.status_code == 400
    assert "Missing transaction identifier" in response.text


def test_handle_login_with_invalid_txn() -> None:
    provider = _make_provider()
    server = FastMCP("Test", auth=provider)
    app = server.http_app(path="/")
    client = TestClient(app)

    response = client.post(
        "/galaxy-auth/login?txn=missing", data={"username": "x", "password": "y"}
    )
    assert response.status_code == 400
    assert "Authorization request is no longer valid" in response.text


@pytest.mark.asyncio()
async def test_handle_login_invalid_credentials(monkeypatch) -> None:
    provider = _make_provider()
    client_info = OAuthClientInformationFull(
        client_id="client-1",
        client_secret="secret",
        redirect_uris=["https://example.com/callback"],
    )
    await provider.register_client(client_info)

    params = AuthorizationParams(
        state="state-123",
        scopes=["galaxy:full"],
        code_challenge="challenge",
        redirect_uri="https://example.com/callback",
        redirect_uri_provided_explicitly=True,
    )
    login_url = await provider.authorize(client_info, params)
    txn = parse_qs(urlparse(login_url).query)["txn"][0]

    server = FastMCP("Test", auth=provider)
    app = server.http_app(path="/")
    client = TestClient(app)

    login_path = urlparse(login_url).path

    with patch.object(
        provider,
        "_get_api_key",
        new=AsyncMock(side_effect=GalaxyAuthenticationError("Invalid Galaxy credentials.")),
    ):
        response = client.post(
            f"{login_path}?txn={txn}",
            data={"username": "user", "password": "wrong"},
        )

    assert response.status_code == 200
    assert "Invalid Galaxy credentials." in response.text


@pytest.mark.asyncio()
async def test_full_authorization_flow(monkeypatch) -> None:
    provider = _make_provider()
    client_info = OAuthClientInformationFull(
        client_id="client-123",
        client_secret="secret",
        redirect_uris=["https://client.example/callback"],
    )
    await provider.register_client(client_info)

    params = AuthorizationParams(
        state="state-token",
        scopes=["galaxy:full"],
        code_challenge="challenge",
        redirect_uri="https://client.example/callback",
        redirect_uri_provided_explicitly=True,
    )
    login_url = await provider.authorize(client_info, params)
    txn = parse_qs(urlparse(login_url).query)["txn"][0]

    server = FastMCP("Test", auth=provider)
    app = server.http_app(path="/")
    client = TestClient(app)

    login_path = urlparse(login_url).path

    with patch.object(
        provider, "_get_api_key", new=AsyncMock(return_value="api-key")
    ), patch.object(
        provider,
        "_get_user_info",
        new=AsyncMock(return_value={"username": "galaxy-user", "email": "user@example.com"}),
    ):
        response = client.post(
            f"{login_path}?txn={txn}",
            data={"username": "user", "password": "pass"},
            follow_redirects=False,
        )

    assert response.status_code == 303
    redirect_url = response.headers["location"]
    parsed = urlparse(redirect_url)
    params_out = parse_qs(parsed.query)
    assert params_out["state"][0] == "state-token"
    auth_code_value = params_out["code"][0]

    auth_code = await provider.load_authorization_code(client_info, auth_code_value)
    assert auth_code is not None

    token = await provider.exchange_authorization_code(client_info, auth_code)
    assert token.access_token

    access_payload = provider.decode_access_token(token.access_token)
    assert access_payload is not None
    assert access_payload["galaxy"]["api_key"] == "api-key"

    refresh = await provider.load_refresh_token(client_info, token.refresh_token)
    assert refresh is not None

    refreshed = await provider.exchange_refresh_token(client_info, refresh, scopes=[])
    assert refreshed.access_token

    loaded_access = await provider.load_access_token(refreshed.access_token)
    assert loaded_access is not None
    assert loaded_access.claims["galaxy_url"] == "https://galaxy.example.com/"


@pytest.mark.asyncio()
async def test_get_api_key_success(monkeypatch) -> None:
    provider = _make_provider()

    class DummyResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"api_key": "generated-key"}

    with patch("galaxy_mcp.auth.requests.get", return_value=DummyResponse()) as mock_get:
        api_key = await provider._get_api_key("user", "password")

    assert api_key == "generated-key"
    mock_get.assert_called_once()


@pytest.mark.asyncio()
async def test_get_api_key_handles_401(monkeypatch) -> None:
    provider = _make_provider()

    class UnauthorizedResponse:
        status_code = 401

        def raise_for_status(self):
            return None

        def json(self):
            return {}

    with patch("galaxy_mcp.auth.requests.get", return_value=UnauthorizedResponse()):
        with pytest.raises(GalaxyAuthenticationError, match="Invalid Galaxy credentials"):
            await provider._get_api_key("user", "password")


@pytest.mark.asyncio()
async def test_get_user_info_success(monkeypatch) -> None:
    provider = _make_provider()

    mock_users = MagicMock()
    mock_users.get_current_user.return_value = {"username": "demo"}
    mock_instance = MagicMock()
    mock_instance.users = mock_users

    with patch("galaxy_mcp.auth.GalaxyInstance", return_value=mock_instance):
        info = await provider._get_user_info("api-key")

    assert info["username"] == "demo"


@pytest.mark.asyncio()
async def test_get_user_info_failure(monkeypatch) -> None:
    provider = _make_provider()

    mock_instance = MagicMock()
    mock_instance.users.get_current_user.side_effect = Exception("Boom")

    with patch("galaxy_mcp.auth.GalaxyInstance", return_value=mock_instance):
        with pytest.raises(GalaxyAuthenticationError, match="Failed to validate API key"):
            await provider._get_user_info("api-key")


def test_render_login_form_includes_error() -> None:
    provider = _make_provider()
    transaction = AuthorizationTransaction(
        client_id="client",
        redirect_uri="https://example.com/callback",
        redirect_uri_provided_explicitly=True,
        state=None,
        code_challenge="challenge",
        code_challenge_method="S256",
        scopes=["galaxy:full"],
        created_at=0.0,
    )

    response = provider._render_login_form(transaction, error="Invalid credentials.")
    assert "Invalid credentials." in response.body.decode("utf-8")
