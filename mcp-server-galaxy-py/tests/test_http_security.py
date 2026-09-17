"""Tests for the HTTP transport guards."""

from unittest.mock import MagicMock, patch

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.testclient import TestClient

from galaxy_mcp import http_security, server
from galaxy_mcp.http_security import (
    HTTPSecurityMiddleware,
    is_loopback_host,
)

from .test_helpers import download_dataset_fn, upload_file_fn


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


class TestLocalFileTools:
    def test_upload_blocked_over_http(self, mock_galaxy_instance, monkeypatch):
        monkeypatch.delenv("GALAXY_MCP_ALLOW_LOCAL_FILES", raising=False)
        with (
            patch.object(http_security, "in_http_request", return_value=True),
            patch("os.path.exists", return_value=True),
            patch.dict(server.galaxy_state, {"connected": True, "gi": mock_galaxy_instance}),
            pytest.raises(ValueError, match="disabled over HTTP"),
        ):
            upload_file_fn("/etc/passwd", "history_1")
        mock_galaxy_instance.tools.upload_file.assert_not_called()

    def test_upload_opt_in_over_http(self, mock_galaxy_instance, monkeypatch):
        monkeypatch.setenv("GALAXY_MCP_ALLOW_LOCAL_FILES", "1")
        with (
            patch.object(http_security, "in_http_request", return_value=True),
            patch("os.path.exists", return_value=True),
            patch.dict(server.galaxy_state, {"connected": True, "gi": mock_galaxy_instance}),
        ):
            upload_file_fn("/data/reads.fastq", "history_1")
        mock_galaxy_instance.tools.upload_file.assert_called_once()

    def test_download_to_path_blocked_over_http(self, mock_galaxy_instance, monkeypatch):
        monkeypatch.delenv("GALAXY_MCP_ALLOW_LOCAL_FILES", raising=False)
        with (
            patch.object(http_security, "in_http_request", return_value=True),
            patch.dict(server.galaxy_state, {"connected": True, "gi": mock_galaxy_instance}),
            pytest.raises(ValueError, match="disabled over HTTP"),
        ):
            download_dataset_fn("dataset_1", file_path="/tmp/out.txt")
        mock_galaxy_instance.datasets.download_dataset.assert_not_called()

    def test_download_to_memory_still_works_over_http(self, mock_galaxy_instance, monkeypatch):
        monkeypatch.delenv("GALAXY_MCP_ALLOW_LOCAL_FILES", raising=False)
        mock_galaxy_instance.datasets.show_dataset.return_value = {
            "id": "dataset_1",
            "name": "out",
            "extension": "txt",
            "state": "ok",
        }
        mock_galaxy_instance.datasets.download_dataset.return_value = b"hello"
        with (
            patch.object(http_security, "in_http_request", return_value=True),
            patch.dict(server.galaxy_state, {"connected": True, "gi": mock_galaxy_instance}),
        ):
            result = download_dataset_fn("dataset_1")
        assert result.success


def test_upload_blocked_over_a_real_http_request(monkeypatch):
    """End to end: the guard has to recognise a genuine streamable-http tool call."""
    monkeypatch.delenv("GALAXY_MCP_ALLOW_LOCAL_FILES", raising=False)
    gi = MagicMock()
    headers = {"accept": "application/json, text/event-stream"}

    with (
        patch.object(server, "auth_provider", None),
        patch.dict(server.galaxy_state, {"connected": True, "gi": gi}),
        patch("os.path.exists", return_value=True),
    ):
        app = server.mcp.http_app(path="/mcp", json_response=True, stateless_http=True)
        with TestClient(app, base_url="http://localhost:8000") as client:
            response = client.post(
                "/mcp",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "upload_file",
                        "arguments": {"path": "/etc/passwd", "history_id": "h1"},
                    },
                },
            )

    assert response.status_code == 200
    body = response.json()
    assert body["result"]["isError"] is True
    assert "disabled over HTTP" in body["result"]["content"][0]["text"]
    gi.tools.upload_file.assert_not_called()


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
