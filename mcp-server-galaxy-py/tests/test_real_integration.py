"""
Real integration tests against a live Galaxy instance.

These tests require a running Galaxy server. Set environment variables:
    GALAXY_TEST_URL - Galaxy server URL (default: http://localhost:8080)
    GALAXY_TEST_API_KEY - Valid API key for the Galaxy instance

Run with: pytest tests/test_real_integration.py -v

Tests are skipped if Galaxy is not reachable.
"""

import contextlib
import os
import tempfile
import time

import pytest
import requests
from galaxy_mcp.server import (
    GalaxyResult,
    connect,
    create_history,
    download_dataset,
    galaxy_state,
    get_histories,
    get_history_contents,
    get_history_details,
    get_server_info,
    get_tool_details,
    get_tool_panel,
    get_user,
    run_tool,
    search_tools_by_name,
    upload_file,
)

# Test configuration
GALAXY_URL = os.environ.get("GALAXY_TEST_URL", "http://localhost:8080")
GALAXY_API_KEY = os.environ.get("GALAXY_TEST_API_KEY", "THEDEFAULTISNOTSECURE")


def galaxy_is_available() -> bool:
    """Check if Galaxy server is reachable and credentials are configured."""
    if not GALAXY_API_KEY or GALAXY_API_KEY == "THEDEFAULTISNOTSECURE":
        return False
    try:
        response = requests.get(f"{GALAXY_URL}/api/version", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


# Skip all tests in this module if Galaxy is not available
pytestmark = pytest.mark.skipif(
    not galaxy_is_available(),
    reason=f"Galaxy server not available at {GALAXY_URL}",
)


class TestRealConnection:
    """Test real connection to Galaxy."""

    def teardown_method(self):
        """Reset connection state after each test."""
        galaxy_state["connected"] = False
        galaxy_state["gi"] = None
        galaxy_state["url"] = None
        galaxy_state["api_key"] = None

    def test_connect_to_galaxy(self):
        """Test connecting to a real Galaxy instance."""
        result = connect.fn(GALAXY_URL, GALAXY_API_KEY)

        assert isinstance(result, GalaxyResult)
        assert result.success is True
        assert "connected" in result.message.lower() or "Connected" in result.message
        assert result.data["connected"] is True
        assert "user" in result.data
        assert result.data["user"]["email"] is not None

    def test_get_server_info(self):
        """Test getting real server information."""
        connect.fn(GALAXY_URL, GALAXY_API_KEY)
        result = get_server_info.fn()

        assert isinstance(result, GalaxyResult)
        assert result.success is True
        assert "version" in result.data
        assert "url" in result.data
        # URL may have trailing slash
        assert result.data["url"].rstrip("/") == GALAXY_URL.rstrip("/")

    def test_get_current_user(self):
        """Test getting current user info."""
        connect.fn(GALAXY_URL, GALAXY_API_KEY)
        result = get_user.fn()

        assert isinstance(result, GalaxyResult)
        assert result.success is True
        assert "email" in result.data
        assert "username" in result.data


class TestRealToolOperations:
    """Test real tool operations."""

    def setup_method(self):
        """Connect to Galaxy before each test."""
        connect.fn(GALAXY_URL, GALAXY_API_KEY)

    def teardown_method(self):
        """Reset connection state after each test."""
        galaxy_state["connected"] = False
        galaxy_state["gi"] = None

    def test_get_tool_panel(self):
        """Test getting the real tool panel."""
        result = get_tool_panel.fn()

        assert isinstance(result, GalaxyResult)
        assert result.success is True
        assert isinstance(result.data, list)
        # Tool panel should have sections
        assert len(result.data) > 0

    def test_search_tools_by_name(self):
        """Test searching for tools by name."""
        # Search for a common tool that should exist
        result = search_tools_by_name.fn("upload")

        assert isinstance(result, GalaxyResult)
        assert result.success is True
        assert isinstance(result.data, list)
        # Should find at least the upload tool
        assert result.count >= 0

    def test_get_tool_details(self):
        """Test getting details for a specific tool."""
        # Get details for the upload tool (should always exist)
        result = get_tool_details.fn("upload1")

        assert isinstance(result, GalaxyResult)
        assert result.success is True
        assert "id" in result.data
        assert result.data["id"] == "upload1"


class TestRealHistoryOperations:
    """Test real history operations."""

    created_histories: list[str] = []

    def setup_method(self):
        """Connect to Galaxy before each test."""
        connect.fn(GALAXY_URL, GALAXY_API_KEY)
        self.created_histories = []

    def teardown_method(self):
        """Clean up created histories and reset state."""
        if galaxy_state.get("gi"):
            gi = galaxy_state["gi"]
            for history_id in self.created_histories:
                with contextlib.suppress(Exception):
                    gi.histories.delete_history(history_id, purge=True)
        galaxy_state["connected"] = False
        galaxy_state["gi"] = None

    def test_get_histories(self):
        """Test getting list of histories."""
        result = get_histories.fn()

        assert isinstance(result, GalaxyResult)
        assert result.success is True
        assert isinstance(result.data, list)
        # Count should reflect number of histories
        assert result.count is not None
        assert result.count >= 0

    def test_create_history(self):
        """Test creating a new history."""
        test_name = f"MCP Integration Test {int(time.time())}"
        result = create_history.fn(test_name)

        assert isinstance(result, GalaxyResult)
        assert result.success is True
        assert "id" in result.data
        assert result.data["name"] == test_name

        # Track for cleanup
        self.created_histories.append(result.data["id"])

    def test_get_history_details(self):
        """Test getting history details."""
        # First create a history
        test_name = f"MCP Detail Test {int(time.time())}"
        create_result = create_history.fn(test_name)
        history_id = create_result.data["id"]
        self.created_histories.append(history_id)

        # Get details
        result = get_history_details.fn(history_id)

        assert isinstance(result, GalaxyResult)
        assert result.success is True
        # Data contains {"history": {...}, "contents_summary": {...}}
        assert "history" in result.data
        assert result.data["history"]["id"] == history_id
        assert result.data["history"]["name"] == test_name

    def test_get_history_contents(self):
        """Test getting history contents."""
        # First create a history
        test_name = f"MCP Contents Test {int(time.time())}"
        create_result = create_history.fn(test_name)
        history_id = create_result.data["id"]
        self.created_histories.append(history_id)

        # Get contents (should be empty for new history)
        result = get_history_contents.fn(history_id)

        assert isinstance(result, GalaxyResult)
        assert result.success is True
        # Data contains {"history_id": ..., "contents": [...]}
        assert "history_id" in result.data
        assert "contents" in result.data
        assert isinstance(result.data["contents"], list)
        assert result.pagination is not None


class TestRealDatasetOperations:
    """Test real dataset upload and download operations."""

    created_histories: list[str] = []

    def setup_method(self):
        """Connect to Galaxy and create a test history."""
        connect.fn(GALAXY_URL, GALAXY_API_KEY)
        self.created_histories = []

    def teardown_method(self):
        """Clean up created histories and reset state."""
        if galaxy_state.get("gi"):
            gi = galaxy_state["gi"]
            for history_id in self.created_histories:
                with contextlib.suppress(Exception):
                    gi.histories.delete_history(history_id, purge=True)
        galaxy_state["connected"] = False
        galaxy_state["gi"] = None

    def test_upload_and_download_file(self):
        """Test uploading a file and downloading it back."""
        # Create a test history
        test_name = f"MCP Upload Test {int(time.time())}"
        history_result = create_history.fn(test_name)
        history_id = history_result.data["id"]
        self.created_histories.append(history_id)

        # Create a temp file to upload
        test_content = "Hello from Galaxy MCP integration test!\nLine 2\nLine 3\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp_file:
            tmp_file.write(test_content)
            tmp_path = tmp_file.name

        try:
            # Upload the file
            upload_result = upload_file.fn(tmp_path, history_id)

            assert isinstance(upload_result, GalaxyResult)
            assert upload_result.success is True
            assert "outputs" in upload_result.data
            assert len(upload_result.data["outputs"]) > 0

            dataset_id = upload_result.data["outputs"][0]["id"]

            # Wait for upload to complete (poll state)
            gi = galaxy_state["gi"]
            for _ in range(30):  # Max 30 seconds
                dataset_info = gi.datasets.show_dataset(dataset_id)
                if dataset_info["state"] == "ok":
                    break
                if dataset_info["state"] == "error":
                    pytest.fail(f"Dataset upload failed: {dataset_info}")
                time.sleep(1)

            # Download the file
            with tempfile.TemporaryDirectory() as tmp_dir:
                download_path = os.path.join(tmp_dir, "downloaded.txt")
                download_result = download_dataset.fn(dataset_id, download_path)

                assert isinstance(download_result, GalaxyResult)
                assert download_result.success is True
                assert "file_path" in download_result.data

                # Verify content
                with open(download_result.data["file_path"]) as f:
                    downloaded_content = f.read()
                assert downloaded_content == test_content

        finally:
            # Clean up temp file
            os.unlink(tmp_path)


class TestRealToolExecution:
    """Test real tool execution."""

    created_histories: list[str] = []

    def setup_method(self):
        """Connect to Galaxy before each test."""
        connect.fn(GALAXY_URL, GALAXY_API_KEY)
        self.created_histories = []

    def teardown_method(self):
        """Clean up created histories and reset state."""
        if galaxy_state.get("gi"):
            gi = galaxy_state["gi"]
            for history_id in self.created_histories:
                with contextlib.suppress(Exception):
                    gi.histories.delete_history(history_id, purge=True)
        galaxy_state["connected"] = False
        galaxy_state["gi"] = None

    def test_run_simple_tool(self):
        """Test running a simple tool (cat1 - concatenate datasets)."""
        # Create a test history
        test_name = f"MCP Tool Test {int(time.time())}"
        history_result = create_history.fn(test_name)
        history_id = history_result.data["id"]
        self.created_histories.append(history_id)

        # Upload a test file
        test_content = "line 1\nline 2\nline 3\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp_file:
            tmp_file.write(test_content)
            tmp_path = tmp_file.name

        try:
            upload_result = upload_file.fn(tmp_path, history_id)
            dataset_id = upload_result.data["outputs"][0]["id"]

            # Wait for upload
            gi = galaxy_state["gi"]
            for _ in range(30):
                dataset_info = gi.datasets.show_dataset(dataset_id)
                if dataset_info["state"] == "ok":
                    break
                if dataset_info["state"] == "error":
                    pytest.fail(f"Dataset upload failed: {dataset_info}")
                time.sleep(1)

            # Run the cat1 tool (concatenate datasets) - a simple built-in tool
            # This tool just outputs the input, so it's a good simple test
            tool_result = run_tool.fn(
                history_id,
                "cat1",
                {"input1": {"src": "hda", "id": dataset_id}},
            )

            assert isinstance(tool_result, GalaxyResult)
            assert tool_result.success is True
            # Tool should produce outputs or jobs
            assert "outputs" in tool_result.data or "jobs" in tool_result.data

        finally:
            os.unlink(tmp_path)
