"""
Test tool-related operations
"""

import asyncio
from unittest.mock import MagicMock, Mock, call, patch

import bioblend
import pytest

from galaxy_mcp.server import _supplies_a_reference
from galaxy_mcp.tool_inputs import check_tool_inputs

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
            assert "(with credentials)" in result.message
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
        # Once for the preflight, once more because naming an input as wrong is not
        # something to do off a cached copy of the tool.
        assert mock_galaxy_instance.tools.show_tool.call_args_list == [
            call("cat1", io_details=True),
            call("cat1", io_details=True),
        ]

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
        """A 404 is NOT treated as input-related -- no error enrichment."""
        mock_galaxy_instance.tools.run_tool.side_effect = bioblend.ConnectionError(
            "Unexpected HTTP status code: 404", body="not found", status_code=404
        )
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError, match="Run tool failed") as exc:
                run_tool_fn("hist1", "cat1", {"input1": {"src": "hda", "id": "d1"}})
        # The preflight fetches the schema before submitting, so show_tool is called;
        # what must not happen is the enrichment, which also pulls a tool test.
        assert "not a sign" not in str(exc.value).lower()
        mock_galaxy_instance.tools.get_tool_tests.assert_not_called()

    def test_run_user_tool_enriches_input_error(self, mock_galaxy_instance):
        """A user tool is not in the toolbox, so the message is built from what it carries."""
        mock_galaxy_instance.url = "http://galaxy/api"
        representation = {
            "class": "GalaxyUserTool",
            "id": "utool",
            "version": "0.1.0",
            "inputs": [{"name": "in", "type": "data"}],
        }
        resolve = type(
            "R",
            (),
            {"json": lambda self: {"tool_id": "utool", "representation": representation}},
        )()
        mock_galaxy_instance.make_get_request.return_value = resolve
        # the job POST fails with a 400
        mock_galaxy_instance.make_post_request.side_effect = bioblend.ConnectionError(
            "Unexpected HTTP status code: 400", body="kwd not provided", status_code=400
        )
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError) as exc:
                run_user_tool_fn("hist1", "uuid-123", {"wrong": "x"})

        msg = str(exc.value)
        assert "not a sign" in msg.lower()
        assert '"in"' in msg or "'in'" in msg
        # Nothing about this tool can be looked up by id, so nothing is.
        mock_galaxy_instance.tools.show_tool.assert_not_called()
        mock_galaxy_instance.tools.get_tool_tests.assert_not_called()

    def test_run_user_tool_error_points_at_a_call_that_can_answer(self, mock_galaxy_instance):
        """With no parameter list to show, the caller still needs somewhere to go."""
        mock_galaxy_instance.url = "http://galaxy/api"
        resolve = type(
            "R",
            (),
            {"json": lambda self: {"tool_id": "utool", "representation": {"version": "0.1.0"}}},
        )()
        mock_galaxy_instance.make_get_request.return_value = resolve
        mock_galaxy_instance.make_post_request.side_effect = bioblend.ConnectionError(
            "Unexpected HTTP status code: 400", body="kwd not provided", status_code=400
        )
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError) as exc:
                run_user_tool_fn("hist1", "uuid-123", {"wrong": "x"})

        msg = str(exc.value)
        assert "list_user_tools()" in msg
        assert "get_tool_details" not in msg
        mock_galaxy_instance.tools.show_tool.assert_not_called()

    def test_run_user_tool_error_names_what_the_representation_proves(self, mock_galaxy_instance):
        """The representation is a parameter list like any other; it can convict a value."""
        mock_galaxy_instance.url = "http://galaxy/api"
        representation = {
            "class": "GalaxyUserTool",
            "id": "utool",
            "version": "0.1.0",
            "inputs": [{"name": "coll", "type": "data_collection"}],
        }
        resolve = type(
            "R",
            (),
            {"json": lambda self: {"tool_id": "utool", "representation": representation}},
        )()
        mock_galaxy_instance.make_get_request.return_value = resolve
        mock_galaxy_instance.make_post_request.side_effect = bioblend.ConnectionError(
            "Unexpected HTTP status code: 400", body="kwd not provided", status_code=400
        )
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError) as exc:
                run_user_tool_fn("hist1", "uuid-123", {"coll": {"src": "dce", "id": "e1"}})

        msg = str(exc.value)
        assert "these are wrong" not in msg  # dce is legitimate on a collection param
        assert '"coll"' in msg or "'coll'" in msg

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


class TestRunToolInputPreflight:
    """run_tool refuses a provable dataset-vs-collection mismatch before submitting."""

    SCHEMA = {
        "id": "cat1",
        "version": "1.0.0",
        "inputs": [{"name": "input1", "type": "data", "multiple": False}],
    }

    def _connected(self, mock_galaxy_instance, schema=None):
        mock_galaxy_instance.tools.show_tool.return_value = schema or self.SCHEMA
        return patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance})

    def test_collection_into_single_data_param_is_not_submitted(self, mock_galaxy_instance):
        with self._connected(mock_galaxy_instance):
            with pytest.raises(ValueError) as exc:
                run_tool_fn("hist1", "cat1", {"input1": {"src": "hdca", "id": "c1"}})

        msg = str(exc.value)
        assert "input1" in msg  # names the parameter
        assert "single dataset" in msg  # what it expects
        assert "collection" in msg  # what it got
        assert "batch" in msg  # the remedy
        assert "get_tool_input_template" in msg  # where to get the shape
        mock_galaxy_instance.tools.run_tool.assert_not_called()

    def test_valid_dataset_still_runs(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.run_tool.return_value = {"jobs": [], "outputs": []}

        with self._connected(mock_galaxy_instance):
            result = run_tool_fn("hist1", "cat1", {"input1": {"src": "hda", "id": "d1"}})

        assert result.success is True
        assert "not pre-checked" not in result.message
        mock_galaxy_instance.tools.run_tool.assert_called_once()

    def test_map_over_still_runs(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.run_tool.return_value = {"jobs": [], "outputs": []}
        value = {"batch": True, "values": [{"src": "hdca", "id": "c1"}]}

        with self._connected(mock_galaxy_instance):
            result = run_tool_fn("hist1", "cat1", {"input1": value})

        assert result.success is True
        mock_galaxy_instance.tools.run_tool.assert_called_once()

    def test_a_scalar_only_run_does_not_fetch_the_schema(self, mock_galaxy_instance):
        """io_details builds the whole tool form server-side; nothing here needs it."""
        mock_galaxy_instance.tools.run_tool.return_value = {"jobs": [], "outputs": []}

        with self._connected(mock_galaxy_instance):
            result = run_tool_fn("hist1", "cat1", {"threshold": 5, "label": "run one"})

        assert result.success is True
        assert "not pre-checked" not in result.message
        mock_galaxy_instance.tools.show_tool.assert_not_called()
        mock_galaxy_instance.tools.run_tool.assert_called_once()

    def test_a_reference_nested_in_a_list_still_fetches_the_schema(self, mock_galaxy_instance):
        """The checker reaches into lists, so the skip must not."""
        mock_galaxy_instance.tools.run_tool.return_value = {"jobs": [], "outputs": []}
        value = [{"src": "hda", "id": "d1"}, {"src": "hda", "id": "d2"}]

        with self._connected(
            mock_galaxy_instance,
            {
                "id": "cat1",
                "version": "1.0.0",
                "inputs": [{"name": "input1", "type": "data", "multiple": True}],
            },
        ):
            result = run_tool_fn("hist1", "cat1", {"input1": value})

        assert result.success is True
        mock_galaxy_instance.tools.show_tool.assert_called_once()

    def test_schema_fetch_failure_runs_anyway_and_says_it_was_not_checked(
        self, mock_galaxy_instance
    ):
        """A broken preflight must not block a run, and must not pretend it checked."""
        mock_galaxy_instance.tools.show_tool.side_effect = Exception("schema boom")
        mock_galaxy_instance.tools.run_tool.return_value = {"jobs": [], "outputs": []}

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = run_tool_fn("hist1", "cat1", {"input1": {"src": "hdca", "id": "c1"}})

        assert result.success is True
        assert "not pre-checked" in result.message
        assert "schema boom" in result.message
        mock_galaxy_instance.tools.run_tool.assert_called_once()

    def test_a_schema_without_a_parameter_list_says_it_was_not_checked(self, mock_galaxy_instance):
        """No inputs key is "could not check", not "checked and fine"."""
        mock_galaxy_instance.tools.run_tool.return_value = {"jobs": [], "outputs": []}

        with self._connected(mock_galaxy_instance, {"id": "cat1", "version": "1.0.0"}):
            result = run_tool_fn("hist1", "cat1", {"input1": {"src": "hdca", "id": "c1"}})

        assert result.success is True
        assert "not pre-checked" in result.message
        assert "without a parameter list" in result.message

    def test_a_tool_with_genuinely_no_inputs_is_checked_quietly(self, mock_galaxy_instance):
        """An empty input list is checkable and fine; it must not warn."""
        mock_galaxy_instance.tools.run_tool.return_value = {"jobs": [], "outputs": []}

        with self._connected(mock_galaxy_instance, {"id": "cat1", "inputs": []}):
            result = run_tool_fn("hist1", "cat1", {})

        assert result.success is True
        assert "not pre-checked" not in result.message

    def test_a_malformed_value_does_not_escape_as_a_traceback(self, mock_galaxy_instance):
        """The checker must never be the thing that stops a submission."""
        schema = {"id": "m", "inputs": [{"name": "q", "type": "data", "multiple": True}]}
        mock_galaxy_instance.tools.run_tool.return_value = {"jobs": [], "outputs": []}

        with self._connected(mock_galaxy_instance, schema):
            result = run_tool_fn("hist1", "m", {"q": [{"src": ["hdca"], "id": "x"}]})

        assert result.success is True
        mock_galaxy_instance.tools.run_tool.assert_called_once()

    def test_deeply_nested_inputs_do_not_escape_as_a_traceback(self, mock_galaxy_instance):
        """The decision to skip the fetch is part of the preflight, so it needs the net too."""
        deep: dict = {}
        cursor = deep
        for _ in range(5000):
            cursor["n"] = {}
            cursor = cursor["n"]
        mock_galaxy_instance.tools.run_tool.return_value = {"jobs": [], "outputs": []}

        with self._connected(mock_galaxy_instance):
            result = run_tool_fn("hist1", "cat1", {"input1": deep})

        assert result.success is True
        assert "not pre-checked" in result.message
        mock_galaxy_instance.tools.run_tool.assert_called_once()

    def test_a_schema_for_another_version_is_not_used_to_reject(self, mock_galaxy_instance):
        """Galaxy can hand back a different installed version than the one asked for."""
        other_version = {
            "id": "toolshed/repos/iuc/fastqc/fastqc/0.74",
            "inputs": [{"name": "input1", "type": "data", "multiple": False}],
        }
        mock_galaxy_instance.tools.run_tool.return_value = {"jobs": [], "outputs": []}

        with self._connected(mock_galaxy_instance, other_version):
            result = run_tool_fn(
                "hist1",
                "toolshed/repos/iuc/fastqc/fastqc/0.73",
                {"input1": {"src": "hdca", "id": "c1"}},
            )

        assert result.success is True
        assert "not pre-checked" in result.message
        assert "0.74" in result.message
        mock_galaxy_instance.tools.run_tool.assert_called_once()

    def test_unversioned_request_resolves_to_the_installed_version(self, mock_galaxy_instance):
        """An unversioned id expanding to a full one is a match, not a mismatch."""
        resolved = {
            "id": "toolshed/repos/iuc/fastqc/fastqc/0.74",
            "inputs": [{"name": "input1", "type": "data", "multiple": False}],
        }

        with self._connected(mock_galaxy_instance, resolved):
            with pytest.raises(ValueError, match="single dataset"):
                run_tool_fn(
                    "hist1",
                    "toolshed/repos/iuc/fastqc/fastqc",
                    {"input1": {"src": "hdca", "id": "c1"}},
                )

        mock_galaxy_instance.tools.run_tool.assert_not_called()

    def test_a_usable_value_at_an_unreachable_repeat_index_is_submitted(self, mock_galaxy_instance):
        """Galaxy drops the instance and runs the job; refusing it would break a run."""
        schema = {
            "id": "cat1",
            "version": "1.0.0",
            "inputs": [
                {
                    "name": "rep",
                    "type": "repeat",
                    "inputs": [{"name": "item", "type": "data", "multiple": False}],
                }
            ],
        }
        mock_galaxy_instance.tools.run_tool.return_value = {"jobs": [], "outputs": []}

        with self._connected(mock_galaxy_instance, schema):
            result = run_tool_fn("hist1", "cat1", {"rep_5|item": {"src": "hda", "id": "d1"}})

        assert result.success is True
        assert "not pre-checked" not in result.message
        mock_galaxy_instance.tools.run_tool.assert_called_once()

    def test_post_hoc_400_names_the_bad_parameter_name(self, mock_galaxy_instance):
        """Galaxy rejected it; say which input looks wrong instead of only dumping."""
        mock_galaxy_instance.tools.run_tool.side_effect = bioblend.ConnectionError(
            "Unexpected HTTP status code: 400", body="kwd not provided", status_code=400
        )
        mock_galaxy_instance.tools.get_tool_tests.return_value = []

        with self._connected(mock_galaxy_instance):
            with pytest.raises(ValueError) as exc:
                run_tool_fn("hist1", "cat1", {"typo_name": {"src": "hda", "id": "d1"}})

        msg = str(exc.value)
        # An unmodelled key is a lead, not a verdict: a key absent from the schema may
        # still be one Galaxy reads itself, so the wording must not call it a typo.
        assert "not in this tool's parameter list" in msg
        assert "outside the tool schema" in msg
        assert "typo_name" in msg
        assert "are wrong" not in msg


class TestRunUserToolInputPreflight:
    """A user tool is scoped to its owner and never enters the global toolbox, so
    the check reads the representation already fetched rather than looking the id
    up in /api/tools, which would 404."""

    REPRESENTATION = {
        "class": "GalaxyUserTool",
        "id": "my_filter",
        "version": "0.1.0",
        "container": "busybox",
        "inputs": [{"name": "in1", "type": "data", "format": "tabular"}],
    }

    def _resolve(self, mock_galaxy_instance, representation=None):
        mock_galaxy_instance.url = "http://galaxy/api"
        response = Mock()
        response.json.return_value = {
            "tool_id": "my_filter",
            "representation": self.REPRESENTATION if representation is None else representation,
        }
        mock_galaxy_instance.make_get_request.return_value = response

    def test_collection_into_single_data_param_is_not_submitted(self, mock_galaxy_instance):
        self._resolve(mock_galaxy_instance)

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError) as exc:
                run_user_tool_fn("hist1", "uuid-1", {"in1": {"src": "hdca", "id": "c1"}})

        msg = str(exc.value)
        assert "in1" in msg
        # Not re-wrapped as "Run user tool failed: ..." -- the preflight message stands.
        assert msg.startswith("Tool inputs failed validation")
        assert "Context: history_id" not in msg
        mock_galaxy_instance.make_post_request.assert_not_called()

    def test_the_check_costs_no_extra_request(self, mock_galaxy_instance):
        """The representation is already in hand; looking the id up would 404."""
        self._resolve(mock_galaxy_instance)
        mock_galaxy_instance.make_post_request.return_value = {"jobs": []}

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = run_user_tool_fn("hist1", "uuid-1", {"in1": {"src": "hda", "id": "d1"}})

        assert result.success is True
        assert "not pre-checked" not in result.message
        mock_galaxy_instance.tools.show_tool.assert_not_called()
        assert mock_galaxy_instance.make_get_request.call_count == 1

    NESTED_REPRESENTATION = {
        "class": "GalaxyUserTool",
        "id": "my_filter",
        "version": "0.1.0",
        "container": "busybox",
        "inputs": [
            {
                "name": "sect",
                "type": "section",
                "parameters": [{"name": "f", "type": "data", "format": ["tabular"]}],
            }
        ],
    }

    def test_a_param_nested_in_the_representation_is_checked_too(self, mock_galaxy_instance):
        """A user tool nests children under `parameters`, not `inputs`."""
        self._resolve(mock_galaxy_instance, representation=self.NESTED_REPRESENTATION)

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError) as exc:
                run_user_tool_fn("hist1", "uuid-1", {"sect|f": {"src": "hdca", "id": "c1"}})

        assert "sect|f" in str(exc.value)
        mock_galaxy_instance.make_post_request.assert_not_called()

    def test_the_refusal_points_at_a_call_that_works_for_a_user_tool(self, mock_galaxy_instance):
        """get_tool_input_template 404s for a tool that is not in the toolbox."""
        self._resolve(mock_galaxy_instance)

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError) as exc:
                run_user_tool_fn("hist1", "uuid-1", {"in1": {"src": "hdca", "id": "c1"}})

        msg = str(exc.value)
        assert "list_user_tools()" in msg
        assert "get_tool_input_template(tool_id) for the expected shape" not in msg

    def test_a_representation_without_inputs_says_it_was_not_checked(self, mock_galaxy_instance):
        self._resolve(mock_galaxy_instance, representation={"version": "0.1.0"})
        mock_galaxy_instance.make_post_request.return_value = {"jobs": []}

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = run_user_tool_fn("hist1", "uuid-1", {"in1": {"src": "hdca", "id": "c1"}})

        assert result.success is True
        assert "not pre-checked" in result.message
        assert "without a parameter list" in result.message
        mock_galaxy_instance.make_post_request.assert_called_once()


class TestARefusalNeverRestsOnTheCache:
    """_TOOL_SCHEMA_CACHE has no expiry, so a stale entry may not block a run.

    A tool upgraded in place keeps the schema this process first saw. Reading a
    message out of that is fine; refusing to submit on it would go on failing a check
    the live definition passes for the life of the process.
    """

    SINGLE = {
        "id": "cat1",
        "version": "1.0.0",
        "inputs": [{"name": "input1", "type": "data", "multiple": False}],
    }
    UPGRADED = {
        "id": "cat1",
        "version": "1.0.0",
        "inputs": [{"name": "input1", "type": "data", "multiple": True}],
    }

    def _prime(self, mock_galaxy_instance):
        """Run once with a dataset so the cache holds the pre-upgrade schema."""
        mock_galaxy_instance.tools.run_tool.return_value = {"jobs": [], "outputs": []}
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            primed = run_tool_fn("hist1", "cat1", {"input1": {"src": "hda", "id": "d1"}})
        assert primed.success is True
        assert mock_galaxy_instance.tools.show_tool.call_count == 1

    def test_a_stale_refusal_is_dropped_once_the_live_schema_is_read(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.show_tool.side_effect = [self.SINGLE, self.UPGRADED]
        self._prime(mock_galaxy_instance)

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = run_tool_fn("hist1", "cat1", {"input1": {"src": "hdca", "id": "c1"}})

        assert result.success is True
        assert "not pre-checked" not in result.message
        assert mock_galaxy_instance.tools.show_tool.call_count == 2
        assert mock_galaxy_instance.tools.run_tool.call_count == 2

    def test_a_refusal_the_live_schema_confirms_still_refuses(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.show_tool.side_effect = [self.SINGLE, self.SINGLE]
        self._prime(mock_galaxy_instance)

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError) as exc:
                run_tool_fn("hist1", "cat1", {"input1": {"src": "hdca", "id": "c1"}})

        assert "input1" in str(exc.value)
        assert mock_galaxy_instance.tools.show_tool.call_count == 2
        assert mock_galaxy_instance.tools.run_tool.call_count == 1

    def test_a_confirmation_that_fails_submits_and_says_it_was_not_checked(
        self, mock_galaxy_instance
    ):
        boom = Exception("schema boom")
        mock_galaxy_instance.tools.show_tool.side_effect = [self.SINGLE, boom]
        self._prime(mock_galaxy_instance)

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = run_tool_fn("hist1", "cat1", {"input1": {"src": "hdca", "id": "c1"}})

        assert result.success is True
        assert "not pre-checked" in result.message
        assert "could not fetch the schema" in result.message
        assert "schema boom" in result.message
        assert mock_galaxy_instance.tools.run_tool.call_count == 2

    def test_a_first_fetch_that_refuses_is_believed_without_a_second_read(
        self, mock_galaxy_instance
    ):
        """Nothing cached means nothing stale; one read is the live definition."""
        mock_galaxy_instance.tools.show_tool.return_value = self.SINGLE
        mock_galaxy_instance.tools.run_tool.return_value = {"jobs": [], "outputs": []}

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError):
                run_tool_fn("hist1", "cat1", {"input1": {"src": "hdca", "id": "c1"}})

        assert mock_galaxy_instance.tools.show_tool.call_count == 1
        mock_galaxy_instance.tools.run_tool.assert_not_called()


class TestTheDiagnosisNeverRestsOnTheCache:
    """Naming an input as wrong is as definitive as refusing it, and gets the same rule.

    The post-hoc path runs after Galaxy has already said 400. If the reason was some
    other parameter, diagnosing off a stale copy of the tool can point the caller at a
    shape that is perfectly correct on the server.
    """

    SINGLE = {
        "id": "cat1",
        "version": "1.0.0",
        "inputs": [
            {"name": "input1", "type": "data", "multiple": False},
            {"name": "count", "type": "integer"},
        ],
    }
    UPGRADED = {
        "id": "cat1",
        "version": "1.0.0",
        "inputs": [
            {"name": "input1", "type": "data", "multiple": True},
            {"name": "count", "type": "integer"},
        ],
    }

    def _rejected_by_galaxy(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.run_tool.side_effect = bioblend.ConnectionError(
            "Unexpected HTTP status code: 400",
            body="Invalid value for count.",
            status_code=400,
        )
        mock_galaxy_instance.tools.get_tool_tests.return_value = []

    def test_the_diagnosis_follows_the_live_definition(self, mock_galaxy_instance):
        """The tool now takes several datasets, so only the integer is really wrong.

        A control rather than a regression test: when the confirmation before the
        submission succeeded, it left the fresh copy behind and the diagnosis was
        already right. The test below is the one that fails without the refetch.
        """
        mock_galaxy_instance.tools.show_tool.side_effect = [self.SINGLE, self.UPGRADED]
        mock_galaxy_instance.tools.run_tool.return_value = {"jobs": [], "outputs": []}
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            run_tool_fn("hist1", "cat1", {"input1": {"src": "hda", "id": "d1"}})

        self._rejected_by_galaxy(mock_galaxy_instance)
        mock_galaxy_instance.tools.show_tool.side_effect = [self.UPGRADED, self.UPGRADED]

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError) as exc:
                run_tool_fn(
                    "hist1", "cat1", {"input1": {"src": "hdca", "id": "c1"}, "count": "many"}
                )

        msg = str(exc.value)
        assert "these are wrong" not in msg
        assert "Invalid value for count" in msg

    def test_nothing_is_named_as_wrong_when_the_tool_cannot_be_read_again(
        self, mock_galaxy_instance
    ):
        """The cached copy may predate an upgrade, so it cannot convict an input."""
        mock_galaxy_instance.tools.show_tool.side_effect = [self.SINGLE, self.UPGRADED]
        mock_galaxy_instance.tools.run_tool.return_value = {"jobs": [], "outputs": []}
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            run_tool_fn("hist1", "cat1", {"input1": {"src": "hda", "id": "d1"}})

        self._rejected_by_galaxy(mock_galaxy_instance)
        mock_galaxy_instance.tools.show_tool.side_effect = Exception("schema boom")

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError) as exc:
                run_tool_fn(
                    "hist1", "cat1", {"input1": {"src": "hdca", "id": "c1"}, "count": "many"}
                )

        msg = str(exc.value)
        assert "these are wrong" not in msg
        assert "Invalid value for count" in msg

    def test_an_unresolved_key_is_still_reported_but_hedged(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.show_tool.return_value = self.SINGLE
        mock_galaxy_instance.tools.run_tool.return_value = {"jobs": [], "outputs": []}
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            run_tool_fn("hist1", "cat1", {"input1": {"src": "hda", "id": "d1"}})

        self._rejected_by_galaxy(mock_galaxy_instance)
        mock_galaxy_instance.tools.show_tool.side_effect = Exception("schema boom")

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError) as exc:
                run_tool_fn("hist1", "cat1", {"input1": {"src": "hda", "id": "d1"}, "typo": 1})

        msg = str(exc.value)
        assert "typo" in msg
        assert "could not be refreshed" in msg


class TestASchemaTheCheckerCannotRead:
    """An exception inside the checker is an unchecked outcome, never an all-clear."""

    BROKEN = {"id": "cat1", "inputs": [{"name": "c", "type": "conditional", "whens": 1}]}

    def test_the_run_goes_ahead_and_says_the_inputs_were_not_checked(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.show_tool.return_value = self.BROKEN
        mock_galaxy_instance.tools.run_tool.return_value = {"jobs": [], "outputs": []}

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = run_tool_fn("hist1", "cat1", {"c|d": {"src": "hdca", "id": "c1"}})

        assert result.success is True
        assert "not pre-checked" in result.message
        assert "could not read the schema" in result.message
        assert "TypeError" in result.message
        mock_galaxy_instance.tools.run_tool.assert_called_once()


class TestOnlyTheSelectedCaseIsChecked:
    """Galaxy populates the selected case and reads nothing under the others."""

    SCHEMA = {
        "id": "cat1",
        "version": "1.0.0",
        "inputs": [
            {
                "name": "c",
                "type": "conditional",
                "test_param": {"name": "mode", "type": "select", "value": "a"},
                "cases": [
                    {"value": "a", "inputs": [{"name": "d", "type": "data", "multiple": False}]},
                    {"value": "b", "inputs": []},
                ],
            }
        ],
    }

    def test_a_value_under_an_unselected_case_is_submitted(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.show_tool.return_value = self.SCHEMA
        mock_galaxy_instance.tools.run_tool.return_value = {"jobs": [], "outputs": []}

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = run_tool_fn(
                "hist1", "cat1", {"c|mode": "b", "c|d": {"src": "hdca", "id": "c1"}}
            )

        assert result.success is True
        assert "not pre-checked" not in result.message
        mock_galaxy_instance.tools.run_tool.assert_called_once()

    BOOLEAN_SCHEMA = {
        "id": "cat1",
        "version": "1.0.0",
        "inputs": [
            {
                "name": "c",
                "type": "conditional",
                "test_param": {
                    "name": "flag",
                    "type": "boolean",
                    "truevalue": "enabled",
                    "falsevalue": "disabled",
                },
                "cases": [
                    {
                        "value": "enabled",
                        "inputs": [{"name": "d", "type": "data", "multiple": False}],
                    },
                    {
                        "value": "disabled",
                        "inputs": [{"name": "d", "type": "data", "multiple": True}],
                    },
                ],
            }
        ],
    }

    def test_a_boolean_selector_does_not_pick_a_case_and_does_not_block(self, mock_galaxy_instance):
        """string_as_bool("enabled") is False, so the literal match is the wrong case."""
        mock_galaxy_instance.tools.show_tool.return_value = self.BOOLEAN_SCHEMA
        mock_galaxy_instance.tools.run_tool.return_value = {"jobs": [], "outputs": []}

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = run_tool_fn(
                "hist1", "cat1", {"c|flag": "enabled", "c|d": {"src": "hdca", "id": "c1"}}
            )

        assert result.success is True
        assert "not pre-checked" not in result.message
        mock_galaxy_instance.tools.run_tool.assert_called_once()

    def test_the_selected_case_is_still_refused(self, mock_galaxy_instance):
        mock_galaxy_instance.tools.show_tool.return_value = self.SCHEMA
        mock_galaxy_instance.tools.run_tool.return_value = {"jobs": [], "outputs": []}

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError) as exc:
                run_tool_fn("hist1", "cat1", {"c|mode": "a", "c|d": {"src": "hdca", "id": "c1"}})

        assert "c|d" in str(exc.value)
        mock_galaxy_instance.tools.run_tool.assert_not_called()


class TestTheSkipNeverHidesAReject:
    """The preflight skips a schema fetch when nothing carries `src`.

    That is only safe if the checker could not have rejected such inputs anyway.
    Rather than argue it, walk every combination of a schema, a key shape and a
    value shape and assert the two never disagree.
    """

    SCHEMAS = [
        {"id": "t", "inputs": [{"name": "p", "type": "data", "multiple": False}]},
        {"id": "t", "inputs": [{"name": "p", "type": "data", "multiple": True}]},
        {"id": "t", "inputs": [{"name": "p", "type": "data_collection"}]},
        {"id": "t", "inputs": [{"name": "p", "type": "text"}]},
        {
            "id": "t",
            "inputs": [
                {
                    "name": "c",
                    "type": "conditional",
                    "cases": [
                        {
                            "value": "a",
                            "inputs": [{"name": "p", "type": "data", "multiple": False}],
                        }
                    ],
                }
            ],
        },
    ]
    VALUES = [
        5,
        "hello",
        None,
        [],
        {},
        [1, 2],
        {"src": "hda", "id": "d"},
        {"src": "hdca", "id": "c"},
        {"src": "ldda", "id": "l"},
        {"src": "dce", "id": "e"},
        [{"src": "hda", "id": "d"}],
        [{"src": "hdca", "id": "c"}],
        [{"src": "hda", "id": "d"}, {"src": "hdca", "id": "c"}],
        {"batch": True, "values": [{"src": "hdca", "id": "c"}]},
        {"nested": {"src": "hda", "id": "d"}},
        {"src": 5, "id": "x"},
        {"id": "x"},
    ]

    def test_a_skipped_check_would_never_have_rejected_anything(self):
        hidden = []
        for schema in self.SCHEMAS:
            for key in ("p", "c|p", "unknown"):
                for value in self.VALUES:
                    inputs = {key: value}
                    if _supplies_a_reference(inputs):
                        continue
                    if check_tool_inputs(schema, inputs)["rejects"]:
                        hidden.append((key, value))

        assert hidden == []

    def test_the_skip_and_the_checker_ask_the_same_question(self, monkeypatch):
        """The sweep above only holds while both sides mean the same by "reference".

        They share one predicate rather than a copy each, which is what makes the
        skip safe. Take the predicate away and both have to fall silent together; if
        either grows its own idea of what a reference is, this fails.
        """
        reference = {"input1": {"src": "hdca", "id": "c1"}}
        schema = {"id": "t", "inputs": [{"name": "input1", "type": "data", "multiple": False}]}

        assert _supplies_a_reference(reference) is True
        assert check_tool_inputs(schema, reference)["rejects"]

        monkeypatch.setattr("galaxy_mcp.server.is_reference", lambda value: False)
        monkeypatch.setattr("galaxy_mcp.tool_inputs.is_reference", lambda value: False)

        assert _supplies_a_reference(reference) is False
        assert check_tool_inputs(schema, reference)["rejects"] == []
