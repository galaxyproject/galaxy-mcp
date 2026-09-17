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
from bioblend.galaxy import GalaxyInstance

from galaxy_mcp.server import GalaxyResult
from tests.mcp_session import LiveMCPSession, ToolCallError

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


# Skip all tests in this module if Galaxy is not available or credentials missing
pytestmark = pytest.mark.skipif(
    not galaxy_is_available(),
    reason=f"Galaxy not available at {GALAXY_URL} or GALAXY_TEST_API_KEY not set",
)


@pytest.fixture(scope="module")
def mcp_session():
    """One client, one session id, held open for the module.

    Per-test isolation still comes from conftest's _reset_galaxy_state, which clears the
    session connection store -- so every test re-connects through its class fixture.
    """
    with LiveMCPSession() as session:
        yield session


@pytest.fixture(scope="module")
def galaxy_client():
    """A plain bioblend client for test bookkeeping (cleanup, polling job state).

    Deliberately not the server's client: bookkeeping should not depend on, or disturb,
    the connection state under test.
    """
    return GalaxyInstance(url=GALAXY_URL, key=GALAXY_API_KEY)


class TestRealConnection:
    """Test real connection to Galaxy."""

    @pytest.fixture(autouse=True)
    def _session(self, mcp_session):
        # No pre-connect here: these tests are what exercises connect().
        self.mcp = mcp_session

    def test_connect_to_galaxy(self):
        """Test connecting to a real Galaxy instance."""
        result = self.mcp.call("connect", GALAXY_URL, GALAXY_API_KEY)

        assert isinstance(result, GalaxyResult)
        assert result.success is True
        # Sessionless connect reports "Validated global Galaxy connection at <url>"; a
        # session reports "... for the current MCP session". Assert the substance rather
        # than one spelling -- pinning "connected" broke silently when the wording moved.
        assert GALAXY_URL.rstrip("/") in result.message
        assert result.data["connected"] is True
        assert "user" in result.data
        assert result.data["user"]["email"] is not None

    def test_get_server_info(self):
        """Test getting real server information."""
        self.mcp.call("connect", GALAXY_URL, GALAXY_API_KEY)
        result = self.mcp.call("get_server_info")

        assert isinstance(result, GalaxyResult)
        assert result.success is True
        assert "version" in result.data
        assert "url" in result.data
        # URL may have trailing slash
        assert result.data["url"].rstrip("/") == GALAXY_URL.rstrip("/")

    def test_get_current_user(self):
        """Test getting current user info."""
        self.mcp.call("connect", GALAXY_URL, GALAXY_API_KEY)
        result = self.mcp.call("get_user")

        assert isinstance(result, GalaxyResult)
        assert result.success is True
        assert "email" in result.data
        assert "username" in result.data


class TestRealToolOperations:
    """Test real tool operations."""

    @pytest.fixture(autouse=True)
    def _session(self, mcp_session):
        self.mcp = mcp_session
        mcp_session.call("connect", GALAXY_URL, GALAXY_API_KEY)

    def test_get_tool_panel(self):
        """Test getting the real tool panel."""
        result = self.mcp.call("get_tool_panel")

        assert isinstance(result, GalaxyResult)
        assert result.success is True
        assert isinstance(result.data, list)
        # Tool panel should have sections
        assert len(result.data) > 0

    def test_search_tools_by_name(self):
        """Test searching for tools by name."""
        # Search for a common tool that should exist
        result = self.mcp.call("search_tools_by_name", "upload")

        assert isinstance(result, GalaxyResult)
        assert result.success is True
        assert isinstance(result.data, list)
        # Should find at least the upload tool
        assert result.count >= 0

    def test_get_tool_details(self):
        """Test getting details for a specific tool."""
        # Get details for the upload tool (should always exist)
        result = self.mcp.call("get_tool_details", "upload1")

        assert isinstance(result, GalaxyResult)
        assert result.success is True
        assert "id" in result.data
        assert result.data["id"] == "upload1"


class TestRealHistoryOperations:
    """Test real history operations."""

    created_histories: list[str] = []

    @pytest.fixture(autouse=True)
    def _session(self, mcp_session, galaxy_client):
        self.mcp = mcp_session
        self.gi = galaxy_client
        self.created_histories = []
        mcp_session.call("connect", GALAXY_URL, GALAXY_API_KEY)
        yield
        for history_id in self.created_histories:
            with contextlib.suppress(Exception):
                galaxy_client.histories.delete_history(history_id, purge=True)

    def test_get_histories(self):
        """Test getting list of histories."""
        result = self.mcp.call("get_histories")

        assert isinstance(result, GalaxyResult)
        assert result.success is True
        assert isinstance(result.data, list)
        # Count should reflect number of histories
        assert result.count is not None
        assert result.count >= 0

    def test_create_history(self):
        """Test creating a new history."""
        test_name = f"MCP Integration Test {int(time.time())}"
        result = self.mcp.call("create_history", test_name)

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
        create_result = self.mcp.call("create_history", test_name)
        history_id = create_result.data["id"]
        self.created_histories.append(history_id)

        # Get details
        result = self.mcp.call("get_history_details", history_id)

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
        create_result = self.mcp.call("create_history", test_name)
        history_id = create_result.data["id"]
        self.created_histories.append(history_id)

        # Get contents (should be empty for new history)
        result = self.mcp.call("get_history_contents", history_id)

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

    @pytest.fixture(autouse=True)
    def _session(self, mcp_session, galaxy_client):
        self.mcp = mcp_session
        self.gi = galaxy_client
        self.created_histories = []
        mcp_session.call("connect", GALAXY_URL, GALAXY_API_KEY)
        yield
        for history_id in self.created_histories:
            with contextlib.suppress(Exception):
                galaxy_client.histories.delete_history(history_id, purge=True)

    def test_upload_and_download_file(self):
        """Test uploading a file and downloading it back."""
        # Create a test history
        test_name = f"MCP Upload Test {int(time.time())}"
        history_result = self.mcp.call("create_history", test_name)
        history_id = history_result.data["id"]
        self.created_histories.append(history_id)

        # Create a temp file to upload
        test_content = "Hello from Galaxy MCP integration test!\nLine 2\nLine 3\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp_file:
            tmp_file.write(test_content)
            tmp_path = tmp_file.name

        try:
            # Upload the file
            upload_result = self.mcp.call("upload_file", tmp_path, history_id)

            assert isinstance(upload_result, GalaxyResult)
            assert upload_result.success is True
            assert "outputs" in upload_result.data
            assert len(upload_result.data["outputs"]) > 0

            dataset_id = upload_result.data["outputs"][0]["id"]

            # Wait for upload to complete (poll state). A shared Galaxy can leave an
            # upload queued well past the old 30s budget; falling through to the download
            # then failed on "state 'queued', not 'ok'", which reads like a download bug.
            # Wait longer, and if it still has not landed say so in those terms.
            gi = self.gi
            upload_timeout = 120
            state = None
            for _ in range(upload_timeout):
                dataset_info = gi.datasets.show_dataset(dataset_id)
                state = dataset_info["state"]
                if state == "ok":
                    break
                if state == "error":
                    pytest.fail(f"Dataset upload failed: {dataset_info}")
                time.sleep(1)
            else:
                pytest.fail(
                    f"Dataset {dataset_id} stuck in state {state!r} after {upload_timeout}s "
                    f"on {GALAXY_URL} -- the server is backed up or the upload is wedged, "
                    "not a download defect."
                )

            # Download the file
            with tempfile.TemporaryDirectory() as tmp_dir:
                download_path = os.path.join(tmp_dir, "downloaded.txt")
                download_result = self.mcp.call("download_dataset", dataset_id, download_path)

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

    @pytest.fixture(autouse=True)
    def _session(self, mcp_session, galaxy_client):
        self.mcp = mcp_session
        self.gi = galaxy_client
        self.created_histories = []
        mcp_session.call("connect", GALAXY_URL, GALAXY_API_KEY)
        yield
        for history_id in self.created_histories:
            with contextlib.suppress(Exception):
                galaxy_client.histories.delete_history(history_id, purge=True)

    def test_run_simple_tool(self):
        """Test running a simple tool (cat1 - concatenate datasets)."""
        # Create a test history
        test_name = f"MCP Tool Test {int(time.time())}"
        history_result = self.mcp.call("create_history", test_name)
        history_id = history_result.data["id"]
        self.created_histories.append(history_id)

        # Upload a test file
        test_content = "line 1\nline 2\nline 3\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp_file:
            tmp_file.write(test_content)
            tmp_path = tmp_file.name

        try:
            upload_result = self.mcp.call("upload_file", tmp_path, history_id)
            dataset_id = upload_result.data["outputs"][0]["id"]

            # Wait for upload
            gi = self.gi
            for _ in range(30):
                dataset_info = gi.datasets.show_dataset(dataset_id)
                if dataset_info["state"] == "ok":
                    break
                if dataset_info["state"] == "error":
                    pytest.fail(f"Dataset upload failed: {dataset_info}")
                time.sleep(1)

            # Run the cat1 tool (concatenate datasets) - a simple built-in tool
            # This tool just outputs the input, so it's a good simple test
            tool_result = self.mcp.call(
                "run_tool",
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


class TestRealIWCOperations:
    """Test real IWC (Intergalactic Workflow Commission) operations.

    These tests fetch workflows from the IWC GitHub repository and can import
    them into a connected Galaxy instance.
    """

    imported_workflows: list[str] = []

    @pytest.fixture(autouse=True)
    def _session(self, mcp_session, galaxy_client):
        self.mcp = mcp_session
        self.gi = galaxy_client
        self.imported_workflows = []
        mcp_session.call("connect", GALAXY_URL, GALAXY_API_KEY)
        yield
        for workflow_id in self.imported_workflows:
            with contextlib.suppress(Exception):
                galaxy_client.workflows.delete_workflow(workflow_id)

    def test_get_iwc_workflows(self):
        """Test fetching all workflows from IWC."""
        result = self.mcp.call("get_iwc_workflows")

        assert isinstance(result, GalaxyResult)
        assert result.success is True
        assert isinstance(result.data, list)
        # IWC should have many workflows
        assert result.count is not None
        assert result.count > 10  # IWC has dozens of workflows

        # Check structure of first workflow
        if result.data:
            workflow = result.data[0]
            assert "trsID" in workflow
            assert "definition" in workflow

    def test_search_iwc_workflows_rna(self):
        """Test searching IWC for RNA-related workflows."""
        result = self.mcp.call("search_iwc_workflows", "rna")

        assert isinstance(result, GalaxyResult)
        assert result.success is True
        assert isinstance(result.data, list)
        # Should find some RNA workflows
        assert result.count is not None
        assert result.count >= 1

        # Check result structure includes enriched fields
        if result.data:
            workflow = result.data[0]
            assert "trsID" in workflow
            assert "name" in workflow
            # New enriched fields
            assert "readme_summary" in workflow
            assert "step_count" in workflow
            assert "authors" in workflow
            assert "categories" in workflow
            assert "tools_used" in workflow
            assert isinstance(workflow["step_count"], int)
            assert isinstance(workflow["authors"], list)
            assert isinstance(workflow["tools_used"], list)

    def test_search_iwc_workflows_assembly(self):
        """Test searching IWC for assembly workflows."""
        result = self.mcp.call("search_iwc_workflows", "assembly")

        assert isinstance(result, GalaxyResult)
        assert result.success is True
        assert isinstance(result.data, list)
        # Should find assembly workflows
        assert result.count >= 1

    def test_search_iwc_workflows_no_results(self):
        """Test searching IWC with a query that returns no results."""
        result = self.mcp.call("search_iwc_workflows", "xyznonexistent123")

        assert isinstance(result, GalaxyResult)
        assert result.success is True
        assert result.count == 0
        assert len(result.data) == 0

    def test_import_workflow_from_iwc(self):
        """Test importing a workflow from IWC into Galaxy."""
        # First, search for a simple workflow to import
        search_result = self.mcp.call("search_iwc_workflows", "quality")

        assert search_result.success is True
        assert search_result.count >= 1

        # Get the trsID of the first matching workflow
        trs_id = search_result.data[0]["trsID"]

        # Import the workflow
        import_result = self.mcp.call("import_workflow_from_iwc", trs_id)

        assert isinstance(import_result, GalaxyResult)
        assert import_result.success is True
        assert "id" in import_result.data
        assert "name" in import_result.data

        # Track for cleanup
        self.imported_workflows.append(import_result.data["id"])

        # Verify the workflow appears in user's workflow list
        workflows_result = self.mcp.call("list_workflows")
        workflow_ids = [w["id"] for w in workflows_result.data]
        assert import_result.data["id"] in workflow_ids

    def test_import_workflow_invalid_trs_id(self):
        """Test importing with an invalid trsID."""
        with pytest.raises(ToolCallError, match="not found in IWC manifest"):
            self.mcp.call("import_workflow_from_iwc", "nonexistent/workflow/id")

    def test_get_iwc_workflow_details(self):
        """Test getting detailed information about a specific IWC workflow."""
        # First, search to get a valid trsID
        search_result = self.mcp.call("search_iwc_workflows", "quality")
        assert search_result.success is True
        assert search_result.count >= 1

        trs_id = search_result.data[0]["trsID"]

        # Get full details
        result = self.mcp.call("get_iwc_workflow_details", trs_id)

        assert isinstance(result, GalaxyResult)
        assert result.success is True

        # Check comprehensive fields
        data = result.data
        assert data["trsID"] == trs_id
        assert "name" in data
        assert "description" in data
        assert "readme" in data  # Full readme, not just summary
        assert "readme_summary" in data
        assert "step_count" in data
        assert "authors" in data
        assert "tools_used" in data
        assert "inputs" in data
        assert "outputs" in data
        assert "license" in data
        assert "categories" in data

        # Full readme should be longer than summary (if readme exists)
        if data["readme"]:
            assert len(data["readme"]) >= len(data["readme_summary"])

        # Inputs and outputs should be lists
        assert isinstance(data["inputs"], list)
        assert isinstance(data["outputs"], list)

    def test_get_iwc_workflow_details_invalid_id(self):
        """Test getting details with an invalid trsID."""
        with pytest.raises(ToolCallError, match="not found in IWC manifest"):
            self.mcp.call("get_iwc_workflow_details", "nonexistent/workflow/id")

    def test_recommend_iwc_workflows_rnaseq(self):
        """Test recommending workflows for RNA-seq analysis."""
        result = self.mcp.call(
            "recommend_iwc_workflows",
            "I have paired-end RNA-seq data and want to do differential expression analysis",
        )

        assert isinstance(result, GalaxyResult)
        assert result.success is True
        assert isinstance(result.data, list)
        assert result.count is not None

        # Should find relevant workflows
        if result.data:
            workflow = result.data[0]
            # Standard enriched fields
            assert "trsID" in workflow
            assert "name" in workflow
            assert "readme_summary" in workflow
            assert "step_count" in workflow
            # Recommendation-specific fields (BM25 score is a float)
            assert "match_score" in workflow
            assert isinstance(workflow["match_score"], float | int)
            assert workflow["match_score"] > 0

    def test_recommend_iwc_workflows_assembly(self):
        """Test recommending workflows for genome assembly."""
        result = self.mcp.call(
            "recommend_iwc_workflows", "assemble bacterial genome nanopore", limit=3
        )

        assert isinstance(result, GalaxyResult)
        assert result.success is True
        assert result.count <= 3  # Respects limit

        if result.data:
            # Results should be sorted by match_score
            scores = [w["match_score"] for w in result.data]
            assert scores == sorted(scores, reverse=True)

    def test_recommend_iwc_workflows_no_matches(self):
        """Test recommending with a query that matches nothing."""
        result = self.mcp.call("recommend_iwc_workflows", "xyznonexistent123 abcfake456")

        assert isinstance(result, GalaxyResult)
        assert result.success is True
        assert result.count == 0
        assert len(result.data) == 0

    def test_recommend_iwc_workflows_with_limit(self):
        """Test that limit parameter is respected."""
        result = self.mcp.call("recommend_iwc_workflows", "sequencing", limit=2)

        assert isinstance(result, GalaxyResult)
        assert result.success is True
        assert len(result.data) <= 2
