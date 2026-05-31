"""
Test tool-related operations
"""

import asyncio
from unittest.mock import MagicMock, patch

import bioblend
import pytest

from .test_helpers import (
    galaxy_state,
    get_tool_input_template_fn,
    get_tool_run_examples_fn,
    run_tool_fn,
    run_user_tool_fn,
    search_tools_fn,
)


class TestToolOperations:
    """Test tool operations"""

    def test_search_tools_fn(self, mock_galaxy_instance):
        """Test tool search functionality"""
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            # Mock get_tools to return all tools (no name parameter)
            mock_galaxy_instance.tools.get_tools.return_value = [
                {"id": "tool1", "name": "Test Tool 1", "description": "Aligns sequences"},
                {"id": "tool2", "name": "Test Tool 2", "description": "Other tool"},
            ]

            # Search with empty query should return all tools
            result = search_tools_fn("")
            assert result.success is True
            assert result.count == 2
            assert len(result.data) == 2
            assert result.data[0]["id"] == "tool1"

            # Search with query should filter by name substring
            result = search_tools_fn("tool 1")
            assert result.success is True
            assert result.count == 1
            assert len(result.data) == 1
            assert result.data[0]["id"] == "tool1"

            # Search should also filter by ID substring
            result = search_tools_fn("tool2")
            assert result.success is True
            assert result.count == 1
            assert len(result.data) == 1
            assert result.data[0]["id"] == "tool2"

    def test_search_tools_with_results(self, mock_galaxy_instance):
        """Test search tools returns filtered results"""
        all_tools = [
            {"id": "tool1", "name": "BWA Aligner", "description": "Aligns sequences"},
            {"id": "tool2", "name": "Samtools", "description": "Process BAM files"},
            {"id": "tool3", "name": "HISAT2", "description": "Fast aligner"},
        ]

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            # Mock get_tools to return all tools
            mock_galaxy_instance.tools.get_tools.return_value = all_tools

            # Search for aligners by name substring
            result = search_tools_fn("align")
            assert result.success is True
            aligners = result.data
            assert len(aligners) == 2
            assert any("BWA" in t["name"] for t in aligners)
            assert any("HISAT2" in t["name"] for t in aligners)

            # Search by ID substring
            result = search_tools_fn("tool1")
            assert result.success is True
            assert len(result.data) == 1
            assert result.data[0]["id"] == "tool1"

    def test_run_tool_fn(self, mock_galaxy_instance):
        """Test running a tool without stored credentials"""
        mock_galaxy_instance.tools.run_tool.return_value = {
            "jobs": [{"id": "job_1", "state": "ok"}],
            "outputs": [{"id": "output_1", "name": "aligned.bam"}],
        }

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            inputs = {"input1": {"src": "hda", "id": "dataset_1"}, "param1": "value1"}

            result = run_tool_fn("test_history_1", "tool1", inputs)

            assert result.success is True
            assert "jobs" in result.data
            assert result.data["jobs"][0]["id"] == "job_1"
            assert "outputs" in result.data
            assert result.data["outputs"][0]["name"] == "aligned.bam"

            mock_galaxy_instance.tools.run_tool.assert_called_once()
            call_args = mock_galaxy_instance.tools.run_tool.call_args
            assert call_args[0] == (
                "test_history_1",
                "tool1",
                {"input1": {"src": "hda", "id": "dataset_1"}, "param1": "value1"},
            )
            assert "credentials_context" in call_args.kwargs
            assert call_args.kwargs["credentials_context"] is None

    def test_run_tool_with_credentials(self, mock_galaxy_instance):
        """Test running a tool with stored credentials"""
        mock_galaxy_instance.users.get_credentials_for_tool.return_value = [
            {
                "user_credentials_id": "cred-1",
                "name": "external_service",
                "version": "1.0",
                "selected_group": {"id": "group-1", "name": "default"},
            }
        ]
        mock_galaxy_instance.tools.run_tool.return_value = {
            "jobs": [{"id": "job_1", "state": "ok"}],
            "outputs": [{"id": "output_1", "name": "aligned.bam"}],
        }

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = run_tool_fn("test_history_1", "tool1", {"param1": "value1"})

            assert result.success is True
            assert result.message.endswith("(with credentials)")
            mock_galaxy_instance.users.get_credentials_for_tool.assert_called_once_with(
                "user1", "tool1"
            )
            mock_galaxy_instance.tools.run_tool.assert_called_once_with(
                "test_history_1",
                "tool1",
                {"param1": "value1"},
                credentials_context=[
                    {
                        "user_credentials_id": "cred-1",
                        "name": "external_service",
                        "version": "1.0",
                        "selected_group": {"id": "group-1", "name": "default"},
                    }
                ],
            )

    def test_run_tool_error(self, mock_galaxy_instance):
        """Test tool execution error handling"""
        mock_galaxy_instance.tools.run_tool.side_effect = Exception("Tool execution failed")

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError, match="Run tool failed"):
                run_tool_fn("test_history_1", "tool1", {})

    def test_run_tool_missing_credentials_error(self, mock_galaxy_instance):
        """Test agent-friendly error when Galaxy requires credentials and none are stored."""
        mock_galaxy_instance.tools.run_tool.side_effect = Exception(
            "Tool execution failed: missing credentials for service"
        )

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError, match="no stored credentials were found"):
                run_tool_fn("test_history_1", "tool1", {})

    def test_run_tool_invalid_stored_credentials_error(self, mock_galaxy_instance):
        """Test agent-friendly error when stored credentials are rejected."""
        mock_galaxy_instance.users.get_credentials_for_tool.return_value = [
            {
                "user_credentials_id": "cred-1",
                "name": "external_service",
                "version": "1.0",
                "selected_group": {"id": "group-1", "name": "default"},
            }
        ]
        mock_galaxy_instance.tools.run_tool.side_effect = Exception(
            "Tool execution failed: invalid user_credentials selection"
        )

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError, match="using stored credentials"):
                run_tool_fn("test_history_1", "tool1", {})

    def test_tool_operations_not_connected(self):
        """Test tool operations fail when not connected"""
        with patch.dict(galaxy_state, {"connected": False}):
            with pytest.raises(Exception):
                search_tools_fn("query")

            with pytest.raises(Exception):
                run_tool_fn("history_1", "tool1", {})

            with pytest.raises(Exception):
                get_tool_run_examples_fn("tool1")

    def test_get_tool_run_examples(self, mock_galaxy_instance):
        """Test retrieving tool usage lessons"""
        mock_galaxy_instance.tools.get_tool_tests.return_value = [
            {
                "name": "Test-1",
                "tool_id": "tool1",
                "tool_version": "1.0",
                "inputs": {"param": ["value"]},
                "outputs": [{"name": "out_file1", "value": "dataset.txt"}],
            }
        ]

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = get_tool_run_examples_fn("tool1", "1.0")

        assert result.success is True
        assert result.count == 1
        assert result.data["requested_version"] == "1.0"
        assert result.data["test_cases"][0]["name"] == "Test-1"
        mock_galaxy_instance.tools.get_tool_tests.assert_called_once_with(
            "tool1", tool_version="1.0"
        )

    def test_get_tool_run_examples_no_version(self, mock_galaxy_instance):
        """Test retrieving tool run examples without specifying version"""
        mock_galaxy_instance.tools.get_tool_tests.return_value = []

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = get_tool_run_examples_fn("tool1")

        assert result.success is True
        assert result.count == 0
        assert result.data["requested_version"] is None
        assert result.data["tool_id"] == "tool1"
        mock_galaxy_instance.tools.get_tool_tests.assert_called_once_with(
            "tool1", tool_version=None
        )

    def test_get_tool_run_examples_uses_request_scoped_gi(self):
        """Must read the per-request gi (state['gi']), not the module global.

        Uses a request-scoped client distinct from galaxy_state['gi'] so the test
        actually fails if the code reverts to reaching for the global client (the
        OAuth regression the request-scoped path guards against).
        """
        request_gi = MagicMock(name="request_gi")
        request_gi.tools.get_tool_tests.return_value = [{"inputs": {"input1": "x"}}]
        global_gi = MagicMock(name="global_gi")

        with (
            patch.dict(galaxy_state, {"connected": True, "gi": global_gi}),
            patch(
                "galaxy_mcp.server._get_request_connection_state",
                return_value={"connected": True, "gi": request_gi},
            ),
        ):
            result = get_tool_run_examples_fn("cat1")

        assert result.success is True
        request_gi.tools.get_tool_tests.assert_called_once_with("cat1", tool_version=None)
        global_gi.tools.get_tool_tests.assert_not_called()

    def test_get_tool_run_examples_error(self, mock_galaxy_instance):
        """Test error handling when fetching tool run lessons fails"""
        mock_galaxy_instance.tools.get_tool_tests.side_effect = Exception("Boom")

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError, match="Get tool run examples failed"):
                get_tool_run_examples_fn("tool1")

    def test_run_tool_enriches_input_error(self, mock_galaxy_instance):
        """A 400 from Galaxy yields a truthful enriched error with the schema."""
        mock_galaxy_instance.tools.run_tool.side_effect = bioblend.ConnectionError(
            "Unexpected HTTP status code: 400",
            body="Required parameter(s) kwd not provided in request.",
            status_code=400,
        )
        mock_galaxy_instance.tools.show_tool.return_value = {
            "id": "cat1",
            "inputs": [{"name": "input1", "type": "data", "optional": False}],
        }
        mock_galaxy_instance.tools.get_tool_tests.return_value = []

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError) as exc:
                run_tool_fn("hist1", "cat1", {"input": {"src": "hda", "id": "d1"}})

        msg = str(exc.value)
        assert "input1" in msg  # real param name surfaced
        assert "not a sign" in msg.lower()  # disclaimer
        assert "kwd" in msg.lower()  # original + wording note preserved
        mock_galaxy_instance.tools.show_tool.assert_called_once_with("cat1", io_details=True)

    def test_run_tool_enrichment_uses_request_scoped_gi(self):
        """Enrichment must fetch the schema via the per-request gi, not the global."""
        request_gi = MagicMock(name="request_gi")
        request_gi.tools.run_tool.side_effect = bioblend.ConnectionError(
            "Unexpected HTTP status code: 400", body="kwd not provided", status_code=400
        )
        request_gi.tools.show_tool.return_value = {
            "id": "cat1",
            "inputs": [{"name": "input1", "type": "data"}],
        }
        request_gi.tools.get_tool_tests.return_value = []
        global_gi = MagicMock(name="global_gi")

        with (
            patch.dict(galaxy_state, {"connected": True, "gi": global_gi}),
            patch(
                "galaxy_mcp.server._get_request_connection_state",
                return_value={"connected": True, "gi": request_gi},
            ),
            pytest.raises(ValueError) as exc,
        ):
            run_tool_fn("hist1", "cat1", {"wrong": "x"})

        assert "input1" in str(exc.value)  # enriched from the request-scoped schema
        request_gi.tools.show_tool.assert_called_once_with("cat1", io_details=True)
        global_gi.tools.show_tool.assert_not_called()

    def test_run_tool_enrichment_applies_through_code_mode_dispatch(self):
        """Code-mode coverage: the run_galaxy_tool meta-tool executes tools via
        ``ctx.fastmcp.call_tool("run_tool", ...)``, so enrichment must survive the
        server dispatch path, not just a direct call to run_tool's function. This
        drives that same dispatch (``mcp.call_tool``) and asserts the enriched error
        comes back instead of the raw 400.
        """
        from fastmcp.exceptions import ToolError

        from galaxy_mcp.server import mcp

        request_gi = MagicMock(name="request_gi")
        request_gi.tools.run_tool.side_effect = bioblend.ConnectionError(
            "Unexpected HTTP status code: 400", body="kwd not provided", status_code=400
        )
        request_gi.tools.show_tool.return_value = {
            "id": "cat1",
            "inputs": [{"name": "input1", "type": "data"}],
        }
        request_gi.tools.get_tool_tests.return_value = []

        async def _dispatch():
            return await mcp.call_tool(
                "run_tool", {"history_id": "h1", "tool_id": "cat1", "inputs": {"wrong": "x"}}
            )

        with patch(
            "galaxy_mcp.server._get_request_connection_state",
            return_value={"connected": True, "gi": request_gi},
        ):
            with pytest.raises(ToolError) as exc:
                asyncio.run(_dispatch())

        msg = str(exc.value)
        assert "input1" in msg  # schema-enriched
        assert "not a sign" in msg.lower()  # disclaimer survived the dispatch

    def test_run_tool_non_input_error_uses_plain_format(self, mock_galaxy_instance):
        """A 404 is NOT treated as input-related -- no schema fetch."""
        mock_galaxy_instance.tools.run_tool.side_effect = bioblend.ConnectionError(
            "Unexpected HTTP status code: 404", body="not found", status_code=404
        )
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError, match="Run tool failed"):
                run_tool_fn("hist1", "cat1", {"input1": {"src": "hda", "id": "d1"}})
        mock_galaxy_instance.tools.show_tool.assert_not_called()

    def test_run_user_tool_enriches_input_error(self, mock_galaxy_instance):
        # resolve uuid -> tool_id succeeds
        mock_galaxy_instance.url = "http://galaxy/api"
        resolve = type(
            "R",
            (),
            {"json": lambda self: {"tool_id": "utool", "representation": {"version": "0.1.0"}}},
        )()
        mock_galaxy_instance.make_get_request.return_value = resolve
        # the job POST fails with a 400
        mock_galaxy_instance.make_post_request.side_effect = bioblend.ConnectionError(
            "Unexpected HTTP status code: 400", body="kwd not provided", status_code=400
        )
        mock_galaxy_instance.tools.show_tool.return_value = {
            "id": "utool",
            "inputs": [{"name": "in", "type": "data"}],
        }
        mock_galaxy_instance.tools.get_tool_tests.return_value = []
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError) as exc:
                run_user_tool_fn("hist1", "uuid-123", {"wrong": "x"})
        assert "not a sign" in str(exc.value).lower()
        assert '"in"' in str(exc.value) or "'in'" in str(exc.value)

    def test_get_tool_input_template(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.show_tool.return_value = {
            "id": "cat1",
            "inputs": [
                {"name": "input1", "type": "data"},
                {
                    "name": "queries",
                    "type": "repeat",
                    "inputs": [{"name": "input2", "type": "data"}],
                },
            ],
        }
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = get_tool_input_template_fn("cat1")
        assert result.success is True
        tmpl = result.data["inputs_template"]
        assert tmpl["input1"] == {"src": "hda", "id": "<dataset_id>"}
        assert tmpl["queries_0|input2"] == {"src": "hda", "id": "<dataset_id>"}
        assert result.data["parameters"][0]["name"] == "input1"
