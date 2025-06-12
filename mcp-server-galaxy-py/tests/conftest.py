"""
Pytest configuration and fixtures for Galaxy MCP server tests
"""

import asyncio
import os
from unittest.mock import Mock, patch

import pytest
from bioblend.galaxy import GalaxyInstance


@pytest.fixture()
def mock_galaxy_instance():
    """Mock GalaxyInstance for tests"""
    mock_gi = Mock(spec=GalaxyInstance)

    # Mock histories
    mock_histories = Mock()
    mock_histories.get_histories.return_value = [
        {"id": "test_history_1", "name": "Test History 1"},
        {"id": "test_history_2", "name": "Test History 2"},
    ]
    mock_histories.show_history.return_value = {
        "id": "test_history_1",
        "name": "Test History 1",
        "state": "ok",
    }
    mock_gi.histories = mock_histories

    # Mock tools
    mock_tools = Mock()
    mock_tools.get_tools.return_value = [
        {"id": "tool1", "name": "Test Tool 1"},
        {"id": "tool2", "name": "Test Tool 2"},
    ]
    mock_gi.tools = mock_tools

    # Mock workflows
    mock_workflows = Mock()
    mock_workflows.get_workflows.return_value = [{"id": "workflow1", "name": "Test Workflow 1"}]
    mock_gi.workflows = mock_workflows

    # Mock invocations
    mock_invocations = Mock()
    mock_invocations.get_invocations.return_value = []
    mock_invocations.show_invocation.return_value = {"id": "inv1", "state": "ok"}
    mock_gi.invocations = mock_invocations

    # Mock datasets
    mock_datasets = Mock()
    mock_datasets.show_dataset.return_value = {"id": "dataset1", "name": "test.txt"}
    mock_datasets.download_dataset.return_value = b"test content"
    mock_gi.datasets = mock_datasets

    # Mock config
    mock_config = Mock()
    mock_config.get_config.return_value = {
        "brand": "Galaxy",
        "logo_url": None,
        "welcome_url": None,
        "support_url": None,
        "citation_url": None,
        "terms_url": None,
        "allow_user_creation": True,
        "allow_user_deletion": False,
        "enable_quotas": True,
        "ftp_upload_site": None,
        "wiki_url": None,
        "screencasts_url": None,
        "library_import_dir": None,
        "user_library_import_dir": None,
        "allow_library_path_paste": False,
        "enable_unique_workflow_defaults": False,
    }
    mock_config.get_version.return_value = {
        "version_major": "23.1",
        "version_minor": "1",
        "extra": {},
    }
    mock_gi.config = mock_config

    # Mock users
    mock_users = Mock()
    mock_users.get_current_user.return_value = {
        "id": "user1",
        "email": "test@example.com",
        "username": "testuser",
    }
    mock_gi.users = mock_users

    return mock_gi


@pytest.fixture(autouse=True)
def _reset_galaxy_state():
    """Reset galaxy state for each test"""
    from galaxy_mcp.server import galaxy_state

    # Save original state
    original_state = galaxy_state.copy()

    # Clear state
    galaxy_state.clear()
    galaxy_state.update({"url": None, "api_key": None, "gi": None, "connected": False})

    yield

    # Restore original state
    galaxy_state.clear()
    galaxy_state.update(original_state)


@pytest.fixture()
def _test_env():
    """Set up test environment variables"""
    original_env = os.environ.copy()

    # Clear Galaxy env variables first
    os.environ.pop("GALAXY_URL", None)
    os.environ.pop("GALAXY_API_KEY", None)

    # Set test values
    os.environ["GALAXY_URL"] = "https://test.galaxy.com"
    os.environ["GALAXY_API_KEY"] = "test_api_key"

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture()
def mcp_server_instance(mock_galaxy_instance, _test_env):
    """Create MCP server instance with mocked Galaxy"""
    # Import and reset galaxy state
    from galaxy_mcp.server import galaxy_state

    # Save original state
    original_state = galaxy_state.copy()

    try:
        with patch("galaxy_mcp.server.GalaxyInstance", return_value=mock_galaxy_instance):
            # Initialize galaxy state
            galaxy_state["gi"] = mock_galaxy_instance
            galaxy_state["connected"] = True
            galaxy_state["url"] = os.environ["GALAXY_URL"]
            galaxy_state["api_key"] = os.environ["GALAXY_API_KEY"]

            from galaxy_mcp.server import mcp

            yield mcp
    finally:
        # Restore original state
        galaxy_state.clear()
        galaxy_state.update(original_state)


@pytest.fixture()
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class MockMCPContext:
    """Mock MCP context for testing tools"""

    def __init__(self, session_data=None):
        self.session_data = session_data or {}
        self.request_id = "test-request-123"
