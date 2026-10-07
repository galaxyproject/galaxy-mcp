"""Regression coverage for destination restrictions and response disclosure."""

import json
from unittest.mock import patch

import pytest
import responses
from fastmcp import Client
from starlette.testclient import TestClient

from galaxy_mcp import server
from galaxy_mcp.client import GalaxyInstance, normalize_galaxy_url

from .test_helpers import connect_fn


@pytest.fixture(autouse=True)
def _configured_destination(monkeypatch):
    monkeypatch.setattr(server, "normalized_galaxy_url", "https://approved.example/galaxy/")
    monkeypatch.setattr(server, "extra_allowed_galaxy_urls", ())
    monkeypatch.setattr(server, "auth_provider", None)
    monkeypatch.setattr(server, "_get_current_session_id", lambda: "security-test")


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/#/",
        "http://127.0.0.1/",
        "http://[::1]/",
        "https://attacker.example/",
        "https://approved.example.attacker.example/galaxy/",
        "https://approved.example@attacker.example/galaxy/",
        "https://approved.example/galaxy/#/",
        "https://approved.example/galaxy/?target=attacker",
        "https://approved.example/other/",
        "https://approved.example:8443/galaxy/",
        "http://approved.example/galaxy/",
        "https://approved.example/galaxy/../other/",
        "",
    ],
)
def test_unapproved_url_rejected_before_client_creation(url):
    with patch.object(server, "GalaxyInstance") as constructor:
        with pytest.raises(ValueError):
            connect_fn(url=url, api_key="dummy-key")
    constructor.assert_not_called()


def test_missing_configuration_rejected(monkeypatch):
    monkeypatch.setattr(server, "normalized_galaxy_url", None)
    with patch.object(server, "GalaxyInstance") as constructor:
        with pytest.raises(ValueError, match="administrator must configure GALAXY_URL"):
            connect_fn(api_key="dummy-key")
    constructor.assert_not_called()


@responses.activate
def test_configured_destination_and_session_credentials(monkeypatch):
    monkeypatch.setenv("GALAXY_URL", "https://attacker.example/")
    responses.get("https://approved.example/galaxy/api/users/current", json={"id": "user-1"})
    result = connect_fn(api_key="user-one-key")
    assert result.success
    assert responses.calls[0].request.headers["x-api-key"] == "user-one-key"
    monkeypatch.setattr(server, "_get_current_session_id", lambda: "second-session")
    connect_fn(api_key="user-two-key")
    assert server._session_connections["security-test"].api_key == "user-one-key"
    assert server._session_connections["second-session"].api_key == "user-two-key"
    assert server.ensure_connected()["url"] == "https://approved.example/galaxy/"


@pytest.mark.parametrize(
    "url", ["https://approved.example/galaxy", "https://approved.example/galaxy/"]
)
@responses.activate
def test_default_destination_is_allowed_explicitly(url):
    responses.get("https://approved.example/galaxy/api/users/current", json={"id": "user-1"})
    assert connect_fn(url=url, api_key="dummy-key").success
    assert len(responses.calls) == 1


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@responses.activate
def test_redirects_are_not_followed(status):
    responses.get(
        "https://approved.example/galaxy/api/users/current",
        status=status,
        headers={"Location": "http://169.254.169.254/latest/meta-data/"},
    )
    with pytest.raises(ValueError, match="Failed to connect"):
        connect_fn(api_key="dummy-key")
    assert len(responses.calls) == 1
    assert "security-test" not in server._session_connections


@responses.activate
def test_extra_destination_uses_own_key_and_session(monkeypatch):
    monkeypatch.setattr(server, "extra_allowed_galaxy_urls", ("https://second.example/galaxy/",))
    monkeypatch.setenv("GALAXY_API_KEY", "operator-key")
    responses.get("https://second.example/galaxy/api/users/current", json={"id": "second-user"})
    result = connect_fn(url="https://second.example/galaxy", api_key="second-key")
    assert result.success
    assert responses.calls[0].request.headers["x-api-key"] == "second-key"
    assert server.ensure_connected()["url"] == "https://second.example/galaxy/"
    assert connect_fn().data["url"] == "https://second.example/galaxy/"
    monkeypatch.setattr(server, "_get_current_session_id", lambda: "another-session")
    assert not server._get_request_connection_state()["connected"]


def test_extra_destination_cannot_inherit_environment_key(monkeypatch):
    monkeypatch.setattr(server, "extra_allowed_galaxy_urls", ("https://second.example/",))
    monkeypatch.setenv("GALAXY_API_KEY", "operator-key")
    with patch.object(server, "GalaxyInstance") as constructor:
        with pytest.raises(ValueError, match="Failed to connect"):
            connect_fn(url="https://second.example/")
    constructor.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "https://second.example/other/",
        "https://second.example/galaxy/#/",
        "https://second.example/galaxy/?url=evil",
        "https://second.example.evil/galaxy/",
        "http://second.example/galaxy/",
        "https://second.example:8443/galaxy/",
    ],
)
def test_extra_destination_must_match_exact_base_url(monkeypatch, url):
    monkeypatch.setattr(server, "extra_allowed_galaxy_urls", ("https://second.example/galaxy/",))
    with patch.object(server, "GalaxyInstance") as constructor:
        with pytest.raises(ValueError):
            connect_fn(url=url, api_key="dummy-key")
    constructor.assert_not_called()


def test_extra_allowlist_cannot_be_changed_by_runtime_environment(monkeypatch):
    monkeypatch.setenv("GALAXY_MCP_EXTRA_ALLOWED_URLS", "https://attacker.example/")
    with pytest.raises(ValueError, match="not allowed"):
        connect_fn(url="https://attacker.example/", api_key="dummy-key")


@responses.activate
def test_get_cannot_enable_redirects():
    responses.get("https://approved.example/", status=302, headers={"Location": "http://other/"})
    client = GalaxyInstance(url="https://approved.example/", key="dummy-key")
    assert (
        client.make_get_request("https://approved.example/", allow_redirects=True).status_code
        == 302
    )
    assert len(responses.calls) == 1


@pytest.mark.parametrize("status", [200, 401, 404, 500])
@responses.activate
def test_connection_error_omits_response_body(status):
    responses.get(
        "https://approved.example/galaxy/api/users/current",
        status=status,
        body="PRIVATE_UPSTREAM_BODY",
    )
    with pytest.raises(ValueError, match="Failed to connect") as error:
        connect_fn(api_key="dummy-key")
    assert "PRIVATE_UPSTREAM_BODY" not in str(error.value)
    assert error.value.__suppress_context__


@pytest.mark.asyncio
async def test_connect_schema_accepts_url_and_api_key():
    async with Client(server.mcp) as client:
        tools = await client.list_tools()
    connect_tool = next(tool for tool in tools if tool.name == "connect")
    assert set(connect_tool.inputSchema["properties"]) == {"url", "api_key"}


def test_disclosure_requests_over_http_cannot_fetch_or_poison_session(monkeypatch):
    monkeypatch.setattr(server, "_get_current_session_id", lambda: server.get_context().session_id)
    headers = {"Accept": "application/json, text/event-stream"}
    with patch("requests.sessions.Session.send") as outgoing:
        with TestClient(server.mcp.http_app(), base_url="http://localhost") as client:
            response = client.post(
                "/mcp",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"roots": {"listChanged": False}, "sampling": {}},
                        "clientInfo": {"name": "test", "version": "1.0.0"},
                    },
                },
            )
            assert response.status_code == 200
            session_id = response.headers["mcp-session-id"]
            headers["mcp-session-id"] = session_id
            client.post(
                "/mcp",
                headers=headers,
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            )
            for target in (
                "https://attacker.example/",
                "http://attacker.example/#/",
                "http://169.254.169.254/latest/meta-data/#/",
                "http://169.254.169.254/latest/meta-data/local-hostname/#/",
            ):
                response = client.post(
                    "/mcp",
                    headers=headers,
                    json={
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "name": "connect",
                            "arguments": {"url": target, "api_key": "true"},
                            "capabilities": {"roots": {"listChanged": False}, "sampling": {}},
                            "clientInfo": {"name": "test", "version": "1.0.0"},
                        },
                    },
                )
                messages = [
                    json.loads(line.removeprefix("data: "))
                    for line in response.text.splitlines()
                    if line.startswith("data: ")
                ]
                assert messages[0]["result"]["isError"]
                assert "Galaxy URL" in response.text
                assert session_id not in server._session_connections

            response = client.post(
                "/mcp",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "search_tools_by_name", "arguments": {"query": "cisa"}},
                },
            )
            assert "Not connected to Galaxy" in response.text
        outgoing.assert_not_called()


@pytest.mark.asyncio
async def test_mcp_tool_errors_do_not_expose_upstream_body(mock_galaxy_instance):
    mock_galaxy_instance.users.get_current_user.side_effect = Exception("PRIVATE_UPSTREAM_BODY")
    with patch.object(server, "GalaxyInstance", return_value=mock_galaxy_instance):
        async with Client(server.mcp) as client:
            result = await client.call_tool(
                "connect", {"api_key": "dummy-key"}, raise_on_error=False
            )
    assert result.is_error
    assert "PRIVATE_UPSTREAM_BODY" not in str(result)


@pytest.mark.parametrize(
    "url",
    [
        "galaxy.example",
        "file:///tmp/data",
        "https://host/#",
        "https://host/?",
        "https://user:pass@host/",
        "https://host/\n",
    ],
)
def test_invalid_administrator_url_rejected(url):
    with pytest.raises(ValueError):
        normalize_galaxy_url(url)
