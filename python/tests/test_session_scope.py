"""Where connect() keeps a caller's Galaxy connection, on each MCP protocol era.

Handshake-era connections carry a session, and connect() stores the connection under it.
The 2026-07-28 era has none: every request gets a fresh session id, so a connection stored
under it is gone by the next call. Over stdio or in-memory there is only one client, so the
server keeps one process-wide connection instead. Over HTTP nothing identifies the client
from one call to the next, so connect(url, api_key) is refused rather than reporting a
success that the next call cannot see.

Each case drives a real client through connect() and then get_user(), the way a caller
does, so a connection that quietly fails to persist shows up as the wrong answer.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from fastmcp import Client
from fastmcp.utilities.tests import asgi_client, asgi_server

from galaxy_mcp import auth as auth_module
from galaxy_mcp import server
from galaxy_mcp.auth import GalaxyOAuthProvider
from galaxy_mcp.client import GalaxyInstance

from .mcp_session import LiveMCPSession

GALAXY_URL = "https://galaxy.example/"
CALLER = {"id": "caller", "username": "caller"}
OPERATOR = {"id": "operator", "username": "operator"}


@pytest.fixture(autouse=True)
def _approved_destination(monkeypatch):
    # Over HTTP connect() only accepts the operator's configured Galaxy.
    monkeypatch.setattr(server, "normalized_galaxy_url", GALAXY_URL)
    monkeypatch.setattr(server, "extra_allowed_galaxy_urls", ())
    monkeypatch.setattr(server, "auth_provider", None)


@pytest.fixture
def caller_galaxy(mock_galaxy_instance):
    mock_galaxy_instance.users.get_current_user.return_value = CALLER
    with patch.object(server, "GalaxyInstance", return_value=mock_galaxy_instance) as made:
        yield made


@pytest.fixture
def operator_galaxy(monkeypatch, mock_galaxy_instance):
    """An operator-configured GALAXY_URL / GALAXY_API_KEY connection, answering as OPERATOR."""
    operator_gi = SimpleNamespace(
        users=SimpleNamespace(get_current_user=lambda: OPERATOR),
        config=mock_galaxy_instance.config,
    )
    server.galaxy_state.update(
        {"url": GALAXY_URL, "api_key": "operator-key", "gi": operator_gi, "connected": True}
    )
    return operator_gi


async def _connect_then_get_user(client: Client) -> tuple[object, object]:
    connected = await client.call_tool_mcp("connect", {"url": GALAXY_URL, "api_key": "caller-key"})
    user = await client.call_tool_mcp("get_user", {})
    return connected, user


def _text(result) -> str:
    return result.content[0].text if result.content else ""


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["legacy", "auto"])
async def test_in_memory_connection_survives_to_the_next_call(caller_galaxy, mode):
    async with Client(server.mcp, mode=mode) as client:
        connected, user = await _connect_then_get_user(client)

    assert not connected.is_error, _text(connected)
    assert not user.is_error, _text(user)
    assert user.structured_content["data"] == CALLER


@pytest.mark.asyncio
async def test_sessionless_in_memory_calls_share_one_process_connection(caller_galaxy):
    async with Client(server.mcp, mode="auto") as client:
        await client.call_tool_mcp("connect", {"url": GALAXY_URL, "api_key": "caller-key"})

    assert list(server._session_connections) == [server._PROCESS_CONNECTION_KEY]


@pytest.mark.asyncio
async def test_handshake_in_memory_connection_is_keyed_on_the_session(caller_galaxy):
    async with Client(server.mcp, mode="legacy") as client:
        await client.call_tool_mcp("connect", {"url": GALAXY_URL, "api_key": "caller-key"})

    assert server._PROCESS_CONNECTION_KEY not in server._session_connections
    assert len(server._session_connections) == 1


def test_the_live_harness_keeps_its_connection(caller_galaxy):
    with LiveMCPSession() as session:
        session.call("connect", url=GALAXY_URL, api_key="caller-key")
        assert session.call("get_user").data == CALLER


@pytest.mark.asyncio
async def test_handshake_http_connection_survives_to_the_next_call(caller_galaxy):
    async with asgi_client(server.mcp, mode="legacy") as client:
        connected, user = await _connect_then_get_user(client)

    assert not connected.is_error, _text(connected)
    assert not user.is_error, _text(user)
    assert user.structured_content["data"] == CALLER


@pytest.mark.asyncio
async def test_sessionless_http_connect_is_refused(caller_galaxy):
    async with asgi_client(server.mcp, mode="auto") as client:
        connected, user = await _connect_then_get_user(client)

    assert connected.is_error
    assert "sessionless 2026-07-28" in _text(connected)
    assert "OAuth" in _text(connected)
    assert "GALAXY_URL and GALAXY_API_KEY" in _text(connected)
    assert user.is_error
    assert "Not connected to Galaxy" in _text(user)
    caller_galaxy.assert_not_called()
    assert server._session_connections == {}


@pytest.mark.asyncio
async def test_sessionless_http_connect_does_not_hand_back_the_operators_galaxy(
    caller_galaxy, operator_galaxy
):
    # Before the refusal, connect() validated the caller's key, kept nothing, and every
    # later call ran against the operator's Galaxy instead.
    async with asgi_client(server.mcp, mode="auto") as client:
        connected = await client.call_tool_mcp(
            "connect", {"url": GALAXY_URL, "api_key": "caller-key"}
        )
        user = await client.call_tool_mcp("get_user", {})

    assert connected.is_error
    assert "sessionless 2026-07-28" in _text(connected)
    caller_galaxy.assert_not_called()
    # The caller was told no, so answering as the operator afterwards is the server's
    # ordinary unauthenticated-HTTP behaviour, not a connection it claimed to have made.
    assert user.structured_content["data"] == OPERATOR


@pytest.mark.asyncio
async def test_sessionless_http_connect_without_arguments_reuses_the_operators_galaxy(
    operator_galaxy,
):
    async with asgi_client(server.mcp, mode="auto") as client:
        connected = await client.call_tool_mcp("connect", {})

    assert not connected.is_error, _text(connected)
    assert connected.structured_content["data"]["auth"] == "global"
    assert connected.structured_content["data"]["user"] == OPERATOR


@pytest.fixture
def oauth_provider(monkeypatch):
    provider = GalaxyOAuthProvider(
        base_url="https://mcp.test",
        galaxy_url=GALAXY_URL,
        session_secret="session-scope-test-secret",
        client_registry_path=None,
    )
    monkeypatch.setattr(auth_module, "_AUTH_PROVIDER", provider)
    monkeypatch.setattr(server, "auth_provider", provider)
    return provider


def _access_token(provider: GalaxyOAuthProvider, username: str) -> str:
    galaxy = {"url": GALAXY_URL, "api_key": f"{username}-key", "username": username}
    issued = provider._issue_tokens(
        client_id=f"{username}-client", scopes=["galaxy:full"], galaxy_payload=galaxy
    )
    return issued.access_token


@pytest.mark.asyncio
async def test_sessionless_http_oauth_resolves_each_request_from_its_token(
    monkeypatch, oauth_provider, operator_galaxy
):
    # The SDK's bearer middleware is what hands get_access_token the request's token; the
    # server here is built without it, so that one hop is stood in for. Everything after
    # it -- decrypting the token and building a client from the key inside -- is real.
    presented: dict[str, str | None] = {"token": None}
    monkeypatch.setattr(
        server,
        "get_access_token",
        lambda: SimpleNamespace(token=presented["token"]) if presented["token"] else None,
    )

    def galaxy_for(*, url, key, user_agent):
        gi = Mock(spec=GalaxyInstance)
        gi.users = Mock()
        gi.users.get_current_user.return_value = {"username": key}
        return gi

    with patch.object(server, "GalaxyInstance", side_effect=galaxy_for):
        async with asgi_client(server.mcp, mode="auto") as client:
            presented["token"] = _access_token(oauth_provider, "alice")
            alice = await client.call_tool_mcp("get_user", {})
            reused = await client.call_tool_mcp("connect", {})
            replaced = await client.call_tool_mcp(
                "connect", {"url": GALAXY_URL, "api_key": "someone-elses-key"}
            )
            presented["token"] = _access_token(oauth_provider, "bob")
            bob = await client.call_tool_mcp("get_user", {})
            presented["token"] = None
            nobody = await client.call_tool_mcp("get_user", {})

    assert alice.structured_content["data"] == {"username": "alice-key"}
    assert bob.structured_content["data"] == {"username": "bob-key"}
    assert reused.structured_content["data"]["auth"] == "oauth"
    assert replaced.is_error
    assert "already signed in through OAuth" in _text(replaced)
    # No token must not fall back to the operator's GALAXY_API_KEY.
    assert nobody.is_error
    assert "Not connected to Galaxy" in _text(nobody)
    assert server._session_connections == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [{}, {"url": "", "api_key": ""}])
async def test_sessionless_http_connect_with_nothing_to_reuse_is_refused(caller_galaxy, arguments):
    async with asgi_client(server.mcp, mode="auto") as client:
        connected = await client.call_tool_mcp("connect", arguments)

    # An empty URL is turned away by URL validation before this check; either way nothing
    # is validated and nothing is kept.
    assert connected.is_error
    caller_galaxy.assert_not_called()
    assert server._session_connections == {}


@pytest.mark.asyncio
async def test_stateless_handshake_http_connect_is_refused(caller_galaxy, operator_galaxy):
    # Without a server-issued session, fastmcp keys a handshake-era request on the client's
    # own mcp-session-id header, or a fresh id per request: connect() used to report
    # success and every later call then ran against the operator's Galaxy.
    async with (
        asgi_server(server.mcp, stateless_http=True) as running,
        running.client(mode="legacy") as client,
    ):
        connected = await client.call_tool_mcp(
            "connect", {"url": GALAXY_URL, "api_key": "caller-key"}
        )

    assert connected.is_error
    assert "stateless HTTP" in _text(connected)
    caller_galaxy.assert_not_called()
    assert server._session_connections == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transport", "http_app_kwargs"),
    [("sse", {}), ("http", {"stateless_http": True})],
)
async def test_no_client_chosen_session_id_reaches_the_process_connection(
    caller_galaxy, transport, http_app_kwargs
):
    # The operator connects over a local sessionless client first, then the same process
    # starts serving HTTP. A remote caller naming any session id it likes, the
    # process connection's printed name included, must not get the operator's Galaxy.
    async with Client(server.mcp, mode="auto") as local:
        await local.call_tool_mcp("connect", {"url": GALAXY_URL, "api_key": "caller-key"})
    assert server._PROCESS_CONNECTION_KEY in server._session_connections

    for chosen in ("process", repr(server._PROCESS_CONNECTION_KEY)):
        async with (
            asgi_server(server.mcp, transport=transport, **http_app_kwargs) as running,
            running.client(mode="legacy", headers={"mcp-session-id": chosen}) as client,
        ):
            user = await client.call_tool_mcp("get_user", {})

        assert user.is_error, _text(user)
        assert "Not connected to Galaxy" in _text(user)


@pytest.mark.asyncio
async def test_building_the_http_app_ends_process_wide_connections(caller_galaxy):
    # Served by `fastmcp run` or mounted in a host app, the process takes remote callers
    # without run_http_server ever being called.
    server.mcp.http_app()
    async with Client(server.mcp, mode="auto") as client:
        connected = await client.call_tool_mcp(
            "connect", {"url": GALAXY_URL, "api_key": "caller-key"}
        )

    assert connected.is_error
    assert server._session_connections == {}


@pytest.mark.asyncio
async def test_sessionless_call_that_may_be_remote_gets_no_process_connection(
    monkeypatch, caller_galaxy
):
    # A process that serves HTTP never treats a call as the operator's, even one whose
    # transport fastmcp did not record.
    monkeypatch.setattr(server, "remote_caller_possible", lambda: True)
    async with Client(server.mcp, mode="auto") as client:
        connected = await client.call_tool_mcp(
            "connect", {"url": GALAXY_URL, "api_key": "caller-key"}
        )

    assert connected.is_error
    assert "sessionless 2026-07-28" in _text(connected)
    assert server._session_connections == {}


@pytest.mark.asyncio
async def test_sessionless_http_transport_alone_is_enough_to_refuse(monkeypatch, caller_galaxy):
    # The transport fastmcp records for the request decides it, even if nothing else marked
    # the process as serving HTTP.
    monkeypatch.setattr(server, "remote_caller_possible", lambda: False)
    async with asgi_client(server.mcp, mode="auto") as client:
        connected = await client.call_tool_mcp(
            "connect", {"url": GALAXY_URL, "api_key": "caller-key"}
        )

    assert connected.is_error
    assert "sessionless 2026-07-28" in _text(connected)
    caller_galaxy.assert_not_called()


@pytest.mark.parametrize(
    ("version", "sessionless"),
    [
        (None, False),
        ("2024-11-05", False),
        ("2025-11-25", False),
        ("2026-07-28", True),
        ("2099-01-01", True),
    ],
)
def test_only_handshake_versions_count_as_having_a_session(version, sessionless):
    request_context = None if version is None else SimpleNamespace(protocol_version=version)
    assert server._is_sessionless(SimpleNamespace(request_context=request_context)) is sessionless
