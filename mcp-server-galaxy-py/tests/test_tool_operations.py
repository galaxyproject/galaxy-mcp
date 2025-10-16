"""
Test tool-related operations
"""

from unittest.mock import patch

import pytest

from .test_helpers import galaxy_state, get_tool_citations_fn, run_tool_fn


class TestToolOperations:
    """Test tool operations"""

    def test_get_tool_citations_scoped_id(self, mock_galaxy_instance):
        """Tool citations should accept scoped IDs returned by search."""
        mock_galaxy_instance.tools.show_tool.return_value = {
            "name": "FastQC",
            "version": "0.72",
            "citations": [{"title": "FastQC citation"}],
        }
        mock_galaxy_instance.tools.show_tool.side_effect = None

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = get_tool_citations_fn("tools:toolshed.g2/fastqc")

        assert result["tool_name"] == "FastQC"
        assert result["tool_version"] == "0.72"
        assert result["citations"] == [{"title": "FastQC citation"}]
        mock_galaxy_instance.tools.show_tool.assert_called_once_with("toolshed.g2/fastqc")

    def test_get_tool_citations_plain_id(self, mock_galaxy_instance):
        """Plain tool identifiers remain supported for backward compatibility."""
        mock_galaxy_instance.tools.show_tool.return_value = {
            "name": "FastQC",
            "version": "0.72",
            "citations": [],
        }
        mock_galaxy_instance.tools.show_tool.side_effect = None

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = get_tool_citations_fn("toolshed.g2/fastqc")

        assert result["tool_name"] == "FastQC"
        mock_galaxy_instance.tools.show_tool.assert_called_once_with("toolshed.g2/fastqc")

    def test_get_tool_citations_invalid_scope(self, mock_galaxy_instance):
        """Invalid scoped IDs should raise a helpful error."""
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError, match="Tool resource identifiers"):
                get_tool_citations_fn("histories:abc123")

    def test_run_tool_fn(self, mock_galaxy_instance):
        """Test running a tool"""
        mock_galaxy_instance.tools.run_tool.return_value = {
            "jobs": [{"id": "job_1", "state": "ok"}],
            "outputs": [{"id": "output_1", "name": "aligned.bam"}],
        }

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            inputs = {"input1": {"src": "hda", "id": "dataset_1"}, "param1": "value1"}

            result = run_tool_fn("test_history_1", "tool1", inputs)

            assert "jobs" in result
            assert result["jobs"][0]["id"] == "job_1"
            assert "outputs" in result
            assert result["outputs"][0]["name"] == "aligned.bam"

            mock_galaxy_instance.tools.run_tool.assert_called_once_with(
                "test_history_1",
                "tool1",
                {"input1": {"src": "hda", "id": "dataset_1"}, "param1": "value1"},
            )

    def test_run_tool_error(self, mock_galaxy_instance):
        """Test tool execution error handling"""
        mock_galaxy_instance.tools.run_tool.side_effect = Exception("Tool execution failed")

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError, match="Failed to run tool"):
                run_tool_fn("test_history_1", "tool1", {})

    def test_tool_operations_not_connected(self):
        """Test tool operations fail when not connected"""
        with patch.dict(galaxy_state, {"connected": False}):
            with pytest.raises(Exception):
                get_tool_citations_fn("tools:abc")

            with pytest.raises(Exception):
                run_tool_fn("history_1", "tool1", {})
