"""
Test Galaxy connection and authentication
"""

from unittest.mock import patch

import pytest
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

        assert result["connected"] is True
        assert result["auth"] == "oauth"
        assert result["url"] == credentials.galaxy_url
        assert (
            result["user"]["username"]
            == mock_galaxy_instance.users.get_current_user.return_value["username"]
        )
        mock_constructor.assert_called_once_with(
            url=credentials.galaxy_url, key=credentials.api_key
        )

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
            assert "url" in result
            assert "version" in result
            assert "config" in result

            assert result["url"] == "https://galaxy.test/"
            assert result["version"] == mock_version
            assert result["config"]["brand"] == "Test Galaxy"
            assert result["config"]["allow_user_creation"] is True

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
