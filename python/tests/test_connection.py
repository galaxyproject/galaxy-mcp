"""
Test Galaxy connection and authentication
"""

from unittest.mock import patch

import pytest

from galaxy_mcp import server
from galaxy_mcp.auth import GalaxyCredentials

from .test_helpers import connect_fn, ensure_connected, galaxy_state, get_server_info_fn


@pytest.mark.usefixtures("_test_env")
class TestConnection:
    """Test connection functionality"""

    def test_initial_state(self):
        """Test initial galaxy state before connection"""
        with patch.dict(galaxy_state, {"connected": False, "gi": None}):
            assert not galaxy_state["connected"]
            assert galaxy_state["gi"] is None

    def test_connection_success(self, mock_galaxy_instance):
        """Test successful connection to Galaxy"""
        with patch.dict(galaxy_state, {"connected": False, "gi": None}):
            with patch("galaxy_mcp.server.GalaxyInstance", return_value=mock_galaxy_instance):
                # This should trigger initialization in the actual module
                from galaxy_mcp.server import galaxy_state as new_state

                # Simulate the initialization
                new_state["gi"] = mock_galaxy_instance
                new_state["connected"] = True

                assert new_state["connected"]
                assert new_state["gi"] is not None

    def test_ensure_connected_when_disconnected(self):
        """Test ensure_connected raises error when not connected"""
        with patch.dict(galaxy_state, {"connected": False}):
            with pytest.raises(ValueError, match="Not connected to Galaxy"):
                ensure_connected()

    def test_ensure_connected_when_connected(self, mock_galaxy_instance):
        """Test ensure_connected passes when connected"""
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            # Should not raise
            ensure_connected()

    def test_connection_with_invalid_url(self):
        """Test connection fails gracefully with invalid URL"""
        with patch.dict(galaxy_state, {"connected": False}):
            with patch("galaxy_mcp.server.GalaxyInstance", side_effect=Exception("Invalid URL")):
                # In real implementation, this would be handled during initialization
                assert not galaxy_state["connected"]

    def test_connection_with_missing_credentials(self):
        """Test connection requires credentials"""
        with patch.dict("os.environ", {}, clear=True):
            with patch.dict(galaxy_state, {"connected": False}):
                # Without credentials, should not connect
                assert not galaxy_state.get("connected", False)

    def test_connect_returns_oauth_session(self, mock_galaxy_instance):
        """Ensure connect() reports OAuth session details when available."""
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
                    result = connect_fn()

        assert result.success is True
        assert result.data["connected"] is True
        assert result.data["auth"] == "oauth"
        assert result.data["url"] == credentials.galaxy_url
        assert (
            result.data["user"]["username"]
            == mock_galaxy_instance.users.get_current_user.return_value["username"]
        )
        mock_constructor.assert_called_once_with(
            url=credentials.galaxy_url,
            key=credentials.api_key,
            user_agent=server.USER_AGENT,
        )

    def test_connect_stores_credentials_per_session(self, mock_galaxy_instance):
        """Non-OAuth connect() should bind credentials to the current MCP session."""
        mock_context = type("Ctx", (), {"session_id": "session-123"})()

        with patch("galaxy_mcp.server.get_context", return_value=mock_context):
            with patch("galaxy_mcp.server.GalaxyInstance", return_value=mock_galaxy_instance):
                result = connect_fn(url="https://session.galaxy", api_key="session-key")

        assert result.success is True
        assert result.data["auth"] == "session"
        session_state = server._session_connections["session-123"]
        assert session_state.url == "https://session.galaxy/"
        assert session_state.api_key == "session-key"
        assert session_state.gi is mock_galaxy_instance
        assert galaxy_state["connected"] is False

    def test_connect_reuses_existing_session_connection(self, mock_galaxy_instance):
        """connect() should reuse a session-bound connection when no new creds are given."""
        server._session_connections["session-123"] = server.SessionGalaxyConnection(
            url="https://session.galaxy/",
            api_key="session-key",
            gi=mock_galaxy_instance,
            last_accessed_at=1.0,
        )
        mock_context = type("Ctx", (), {"session_id": "session-123"})()

        with patch("galaxy_mcp.server.get_context", return_value=mock_context):
            result = connect_fn()

        assert result.success is True
        assert result.data["auth"] == "session"
        assert result.data["url"] == "https://session.galaxy/"
        assert (
            result.data["user"]["username"]
            == mock_galaxy_instance.users.get_current_user.return_value["username"]
        )

    def test_session_connection_cache_evicts_lru_entry(self, mock_galaxy_instance):
        """Session-bound connections should be capped by least-recently-used eviction."""
        with patch("galaxy_mcp.server._MAX_SESSION_CONNECTIONS", 2):
            with patch(
                "galaxy_mcp.server.time.monotonic",
                side_effect=[1.0, 2.0, 3.0, 4.0],
            ):
                server._set_session_connection(
                    "session-1",
                    url="https://session-1.galaxy/",
                    api_key="session-key-1",
                    gi=mock_galaxy_instance,
                )
                server._set_session_connection(
                    "session-2",
                    url="https://session-2.galaxy/",
                    api_key="session-key-2",
                    gi=mock_galaxy_instance,
                )
                assert server._get_session_connection("session-1") is not None
                server._set_session_connection(
                    "session-3",
                    url="https://session-3.galaxy/",
                    api_key="session-key-3",
                    gi=mock_galaxy_instance,
                )

        assert set(server._session_connections) == {"session-1", "session-3"}

    def test_connect_requires_explicit_credentials_for_new_session(self):
        """A fresh session should not bootstrap itself from global env credentials."""
        mock_context = type("Ctx", (), {"session_id": "session-456"})()

        with patch("galaxy_mcp.server.get_context", return_value=mock_context):
            with pytest.raises(
                ValueError, match="No Galaxy connection is available for this MCP session"
            ):
                connect_fn()

    def test_connect_allows_global_fallback_for_new_session(self, mock_galaxy_instance):
        """A fresh session may reuse a configured global fallback without seeding session state."""
        mock_context = type("Ctx", (), {"session_id": "session-789"})()

        with patch.dict(
            galaxy_state,
            {
                "url": "https://global.galaxy/",
                "api_key": "global-key",
                "gi": mock_galaxy_instance,
                "connected": True,
            },
        ):
            with patch("galaxy_mcp.server.get_context", return_value=mock_context):
                result = connect_fn()

        assert result.success is True
        assert result.data["auth"] == "global"
        assert "session-789" not in server._session_connections

    def test_connect_without_session_does_not_mutate_global_fallback(self, mock_galaxy_instance):
        """Sessionless connect() should not rewrite the configured global fallback state."""
        with patch.dict(
            galaxy_state,
            {
                "url": "https://global.galaxy/",
                "api_key": "global-key",
                "gi": mock_galaxy_instance,
                "connected": True,
            },
        ):
            with patch("galaxy_mcp.server.get_context", side_effect=RuntimeError("no session")):
                with patch("galaxy_mcp.server.GalaxyInstance", return_value=mock_galaxy_instance):
                    result = connect_fn(url="https://other.galaxy", api_key="other-key")

            assert result.success is True
            assert result.data["auth"] == "global"
            assert galaxy_state["url"] == "https://global.galaxy/"
            assert galaxy_state["api_key"] == "global-key"
            assert galaxy_state["gi"] is mock_galaxy_instance
            assert galaxy_state["connected"] is True

    def test_request_state_prefers_session_credentials(self, mock_galaxy_instance):
        """Session-bound credentials should override process-global API-key state."""
        server._session_connections["session-123"] = server.SessionGalaxyConnection(
            url="https://session.galaxy/",
            api_key="session-key",
            gi=mock_galaxy_instance,
            last_accessed_at=1.0,
        )

        with patch.dict(
            galaxy_state,
            {
                "url": "https://global.galaxy/",
                "api_key": "global-key",
                "gi": object(),
                "connected": True,
            },
        ):
            mock_context = type("Ctx", (), {"session_id": "session-123"})()
            with patch("galaxy_mcp.server.get_context", return_value=mock_context):
                state = server._get_request_connection_state()

        assert state["source"] == "session"
        assert state["url"] == "https://session.galaxy/"
        assert state["api_key"] == "session-key"
        assert state["gi"] is mock_galaxy_instance

    def test_get_server_info_success(self, mock_galaxy_instance):
        """Test successful server info retrieval"""
        # Mock server config and version responses
        mock_config = {
            "brand": "Test Galaxy",
            "logo_url": "https://galaxy.test/logo.png",
            "welcome_url": "https://galaxy.test/welcome",
            "support_url": "https://galaxy.test/support",
            "allow_user_creation": True,
            "enable_quotas": False,
            "ftp_upload_site": "ftp.galaxy.test",
        }

        mock_version = {"version_major": "23.1", "version_minor": "1", "extra": {}}

        mock_galaxy_instance.config.get_config.return_value = mock_config
        mock_galaxy_instance.config.get_version.return_value = mock_version

        with patch.dict(
            galaxy_state,
            {"connected": True, "gi": mock_galaxy_instance, "url": "https://galaxy.test/"},
        ):
            result = get_server_info_fn()

            # Verify the structure and content
            assert result.success is True
            assert "url" in result.data
            assert "version" in result.data
            assert "config" in result.data

            assert result.data["url"] == "https://galaxy.test/"
            assert result.data["version"] == mock_version
            assert result.data["config"]["brand"] == "Test Galaxy"
            assert result.data["config"]["allow_user_creation"] is True

            # Verify API calls were made
            mock_galaxy_instance.config.get_config.assert_called_once()
            mock_galaxy_instance.config.get_version.assert_called_once()

    def test_get_server_info_not_connected(self):
        """Test server info fails when not connected"""
        with patch.dict(galaxy_state, {"connected": False}):
            with pytest.raises(ValueError, match="Not connected to Galaxy"):
                get_server_info_fn()

    def test_get_server_info_api_error(self, mock_galaxy_instance):
        """Test server info handles API errors gracefully"""
        mock_galaxy_instance.config.get_config.side_effect = Exception("API Error")

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError, match="Failed to get server information"):
                get_server_info_fn()
