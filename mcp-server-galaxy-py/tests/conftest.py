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
    histories_data = [
        {"id": "test_history_1", "name": "Test History 1", "deleted": False},
        {"id": "test_history_2", "name": "Test History 2", "deleted": False},
    ]

    def mock_get_histories(limit=None, deleted=False, **kwargs):
        if deleted:
            return [{"id": "deleted_history", "name": "Deleted History", "deleted": True}]
        return histories_data[: limit or len(histories_data)]

    def mock_show_history(history_id, contents=False, types=None, **kwargs):
        if contents and types == ["dataset_collection"]:
            return [
                {
                    "id": "dc1",
                    "name": "Test Collection",
                    "collection_type": "list",
                    "element_count": 2,
                    "deleted": False,
                }
            ]
        return {
            "id": history_id,
            "name": next((h["name"] for h in histories_data if h["id"] == history_id), history_id),
            "state": "ok",
            "tags": ["history"],
            "size": 12345,
            "deleted": False,
        }

    mock_histories.get_histories.side_effect = mock_get_histories
    mock_histories.show_history.side_effect = mock_show_history
    mock_gi.histories = mock_histories

    # Mock tools
    mock_tools = Mock()

    tool_catalog = {
        "tool1": {
            "name": "Test Tool 1",
            "description": "General purpose analysis tool",
            "tool_type": "default",
            "labels": ["test"],
            "versions": {
                "2.0": {
                    "id": "tool1/2.0",
                    "citations": [{"title": "Tool 1 reference"}],
                },
                "1.0": {
                    "id": "tool1/1.0",
                    "citations": [],
                },
            },
            "latest": "2.0",
        },
        "tool2": {
            "name": "RNA Analysis Tool",
            "description": "RNA analysis",
            "tool_type": "default",
            "labels": ["rna"],
            "versions": {
                "1.0": {
                    "id": "tool2/1.0",
                    "citations": [],
                }
            },
            "latest": "1.0",
        },
    }

    mock_tools.get_tools.return_value = [
        {
            "id": version_data["id"],
            "name": catalog["name"],
            "description": catalog["description"],
            "version": version,
            "tool_type": catalog["tool_type"],
            "labels": catalog["labels"],
        }
        for base_id, catalog in tool_catalog.items()
        for version, version_data in catalog["versions"].items()
    ]

    def _split_tool_id(tool_id: str) -> tuple[str, str | None]:
        if "/" not in tool_id:
            return tool_id, None
        base, suffix = tool_id.rsplit("/", 1)
        if not suffix:
            return base, None
        return base, suffix

    def mock_show_tool(tool_id, io_details=False, tool_version=None, **kwargs):
        base_id, version_suffix = _split_tool_id(tool_id)
        if base_id not in tool_catalog:
            base_id = tool_id
            version_suffix = None
        if base_id not in tool_catalog:
            raise ValueError(f"Unknown tool id {tool_id}")

        catalog = tool_catalog[base_id]
        version = tool_version or version_suffix or catalog["latest"]
        if version not in catalog["versions"]:
            raise ValueError(f"Unknown version {version} for tool {base_id}")

        versions_listing = [
            {"id": data["id"], "version": ver}
            for ver, data in sorted(catalog["versions"].items(), reverse=True)
        ]

        version_data = catalog["versions"][version]
        return {
            "id": version_data["id"],
            "tool_id": base_id,
            "name": catalog["name"],
            "description": catalog["description"],
            "version": version,
            "versions": versions_listing,
            "citations": version_data.get("citations", []),
        }

    mock_tools.show_tool.side_effect = mock_show_tool
    mock_gi.tools = mock_tools

    # Mock workflows
    mock_workflows = Mock()
    mock_workflows.get_workflows.return_value = [
        {"id": "workflow1", "name": "Test Workflow 1", "annotation": "RNA workflow"}
    ]
    mock_gi.workflows = mock_workflows

    # Mock invocations
    mock_invocations = Mock()
    mock_invocations.get_invocations.return_value = [
        {
            "id": "inv1",
            "workflow_id": "workflow1",
            "workflow_name": "Test Workflow 1",
            "state": "scheduled",
            "history_id": "test_history_1",
        }
    ]
    mock_invocations.show_invocation.return_value = {
        "id": "inv1",
        "state": "ok",
        "workflow_id": "workflow1",
    }
    mock_gi.invocations = mock_invocations

    # Mock datasets
    mock_datasets = Mock()
    mock_datasets.get_datasets.return_value = [
        {
            "id": "dataset1",
            "name": "Test Dataset Alpha",
            "extension": "txt",
            "state": "ok",
            "history_id": "test_history_1",
            "deleted": False,
            "visible": True,
        }
    ]
    mock_datasets.show_dataset.return_value = {"id": "dataset1", "name": "test.txt", "state": "ok"}
    mock_datasets.download_dataset.return_value = b"test content"
    mock_gi.datasets = mock_datasets

    # Mock dataset collections
    mock_dataset_collections = Mock()
    mock_dataset_collections.show_dataset_collection.return_value = {
        "id": "dc1",
        "name": "Test Collection",
        "elements": [],
    }
    mock_gi.dataset_collections = mock_dataset_collections

    # Mock libraries
    mock_libraries = Mock()
    mock_libraries.get_libraries.return_value = [
        {"id": "lib1", "name": "Test Library", "description": "Shared datasets", "deleted": False}
    ]

    def mock_show_library(library_id, contents=False):
        if contents:
            return [
                {
                    "id": "ld1",
                    "name": "Test Library Dataset",
                    "type": "file",
                    "deleted": False,
                }
            ]
        return {"id": library_id, "name": "Test Library", "description": "Shared datasets"}

    mock_libraries.show_library.side_effect = mock_show_library
    mock_libraries.show_dataset.return_value = {
        "id": "ld1",
        "name": "Test Library Dataset",
        "library_id": "lib1",
    }
    mock_gi.libraries = mock_libraries

    # Mock jobs
    mock_jobs = Mock()
    mock_jobs.get_jobs.return_value = [
        {
            "id": "job1",
            "tool_id": "tool1",
            "state": "ok",
            "history_id": "test_history_1",
            "exit_code": 0,
        }
    ]
    mock_jobs.show_job.return_value = {"id": "job1", "state": "ok", "tool_id": "tool1"}
    mock_gi.jobs = mock_jobs

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
    os.environ.pop("GALAXY_MCP_PUBLIC_URL", None)
    os.environ.pop("GALAXY_MCP_SESSION_SECRET", None)

    # Set test values
    os.environ["GALAXY_URL"] = "https://test.galaxy.com"
    os.environ["GALAXY_MCP_PUBLIC_URL"] = "https://mcp.test"

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
            galaxy_state["api_key"] = "test_api_key"

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
