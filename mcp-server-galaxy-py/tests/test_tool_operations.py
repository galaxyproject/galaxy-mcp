"""
Test tool-related operations
"""

from unittest.mock import patch

import pytest

from .test_helpers import fetch_fn, galaxy_state, run_tool_fn


class TestToolOperations:
    """Test tool operations"""

    def test_fetch_tool_includes_citations(self, mock_galaxy_instance):
        """Fetching a tool should include aggregated citations."""
        mock_galaxy_instance.tools.show_tool.side_effect = [
            {
                "id": "toolshed.g2/fastqc/0.72",
                "name": "FastQC",
                "version": "0.72",
                "citations": [{"title": "FastQC citation"}],
                "versions": [{"id": "toolshed.g2/fastqc/0.71"}],
            },
            {
                "id": "toolshed.g2/fastqc/0.71",
                "name": "FastQC",
                "version": "0.71",
                "citations": [{"title": "Older citation"}],
                "versions": [],
            },
        ]

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = fetch_fn("tools:toolshed.g2/fastqc/0.72")

        metadata = result["metadata"]
        assert metadata["tool_id"] == "toolshed.g2/fastqc"
        assert metadata["citations"] == [
            {"title": "FastQC citation"},
            {"title": "Older citation"},
        ]
        assert len(metadata["versions"]) == 2
        mock_galaxy_instance.tools.show_tool.assert_any_call("toolshed.g2/fastqc", io_details=True)
        mock_galaxy_instance.tools.show_tool.assert_any_call(
            "toolshed.g2/fastqc", io_details=True, tool_version="0.71"
        )

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
                fetch_fn("tools:abc")

            with pytest.raises(Exception):
                run_tool_fn("history_1", "tool1", {})
