"""Tests for the HTTP transport guards."""

from unittest.mock import patch

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.testclient import TestClient

from galaxy_mcp import server
from galaxy_mcp.http_security import (
    HTTPSecurityMiddleware,
    is_loopback_host,
)


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", True),
        ("127.8.8.8", True),
        ("localhost", True),
        ("::1", True),
        ("[::1]", True),
        ("0.0.0.0", False),
        ("::", False),
        ("192.168.1.20", False),
        ("mcp.example.com", False),
    ],
)
def test_is_loopback_host(host, expected):
    assert is_loopback_host(host) is expected


def _client(**middleware_kwargs) -> TestClient:
    app = Starlette()

    async def ping(_request):
        return PlainTextResponse("pong")

    app.add_route("/ping", ping, methods=["GET", "POST"])
    app.add_middleware(HTTPSecurityMiddleware, **middleware_kwargs)
    return TestClient(app, base_url="http://localhost:8000")


class TestHostCheck:
    def test_loopback_host_is_served(self):
        assert _client(auth_enabled=False).get("/ping").status_code == 200

    def test_rebound_host_is_rejected(self):
        response = _client(auth_enabled=False).get("/ping", headers={"host": "evil.example:8000"})
        assert response.status_code == 421

    def test_configured_host_is_served(self):
        client = _client(auth_enabled=False, allowed_hosts={"mcp.lab.internal"})
        assert client.get("/ping", headers={"host": "mcp.lab.internal:8000"}).status_code == 200

    def test_wildcard_disables_the_check(self):
        client = _client(auth_enabled=False, allowed_hosts={"*"})
        assert client.get("/ping", headers={"host": "anything.example"}).status_code == 200

    def test_oauth_deployments_are_not_host_checked(self):
        client = _client(auth_enabled=True)
        assert client.get("/ping", headers={"host": "mcp.example.com"}).status_code == 200


class TestOriginCheck:
    def test_requests_without_origin_pass_through(self):
        response = _client(auth_enabled=False).post("/ping")
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers

    def test_foreign_origin_is_rejected_without_auth(self):
        client = _client(auth_enabled=False)
        response = client.post("/ping", headers={"origin": "https://evil.example"})
        assert response.status_code == 403
        assert "access-control-allow-origin" not in response.headers

    def test_foreign_preflight_is_rejected_without_auth(self):
        client = _client(auth_enabled=False)
        response = client.options(
            "/ping",
            headers={"origin": "https://evil.example", "access-control-request-method": "POST"},
        )
        assert response.status_code == 403

    def test_local_pages_are_allowed_without_auth(self):
        client = _client(auth_enabled=False)
        response = client.post("/ping", headers={"origin": "http://localhost:6274"})
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://localhost:6274"

    def test_allowlisted_origin_is_allowed(self):
        client = _client(auth_enabled=False, allowed_origins={"https://app.example.org"})
        response = client.options(
            "/ping",
            headers={
                "origin": "https://app.example.org",
                "access-control-request-method": "POST",
            },
        )
        assert response.status_code == 204
        assert response.headers["access-control-allow-origin"] == "https://app.example.org"

    def test_oauth_answers_any_origin_by_default(self):
        client = _client(auth_enabled=True)
        response = client.options("/ping", headers={"origin": "https://client.example"})
        assert response.status_code == 204

    def test_oauth_respects_an_allowlist_when_given(self):
        client = _client(auth_enabled=True, allowed_origins={"https://app.example.org"})
        response = client.post("/ping", headers={"origin": "https://client.example"})
        assert response.status_code == 403


class TestHeaderEdgeCases:
    @pytest.mark.parametrize("host", ["127.0.0.2:8000", "localhost.:8000", "[::1]:8000"])
    def test_every_loopback_spelling_is_served(self, host):
        assert _client(auth_enabled=False).get("/ping", headers={"host": host}).status_code == 200

    @pytest.mark.parametrize(
        "host", ["localhost.evil.example", "127.0.0.1.evil.example", "localhost@evil.example"]
    )
    def test_lookalike_hosts_are_rejected(self, host):
        assert _client(auth_enabled=False).get("/ping", headers={"host": host}).status_code == 421

    def test_malformed_host_is_rejected_not_crashed(self):
        assert _client(auth_enabled=False).get("/ping", headers={"host": "[::1"}).status_code == 421

    @pytest.mark.parametrize("origin", ["null", "http://[::1", "http://localhost.evil.example"])
    def test_bad_origins_are_rejected_not_crashed(self, origin):
        response = _client(auth_enabled=False).post("/ping", headers={"origin": origin})
        assert response.status_code == 403


def test_public_url_host_is_not_trusted_when_oauth_is_off():
    """GALAXY_MCP_PUBLIC_URL set but OAuth not running must not open the Host check."""
    with (
        patch.object(server, "auth_provider", None),
        patch.object(server, "public_base_url", "https://mcp.example.com"),
    ):
        app = server.mcp.http_app(path="/mcp", json_response=True, stateless_http=True)
        with TestClient(app, base_url="http://localhost:8000") as client:
            response = client.post("/mcp", headers={"host": "mcp.example.com"}, json={})
    assert response.status_code == 421
