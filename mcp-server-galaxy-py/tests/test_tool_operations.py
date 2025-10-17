"""
Test tool-related operations
"""

from unittest.mock import patch

import pytest

from .test_helpers import galaxy_state, run_tool_fn, search_tools_fn


class TestToolOperations:
    """Test tool operations"""

    def test_search_tools_includes_citations(self, mock_galaxy_instance):
        """Tool search should include aggregated citations."""
        mock_galaxy_instance.tools.get_tools.return_value = [
            {
                "id": "toolshed.g2/fastqc/0.72",
                "name": "FastQC",
                "description": "QC tool",
                "version": "0.72",
                "tool_type": "default",
                "labels": ["qc"],
            }
        ]

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}), patch(
            "galaxy_mcp.server._collect_tool_details",
            return_value={
                "tool_id": "toolshed.g2/fastqc",
                "citations": [
                    {"title": "FastQC citation"},
                    {"title": "Older citation"},
                ],
                "versions": [
                    {
                        "id": "toolshed.g2/fastqc/0.72",
                        "version": "0.72",
                        "resource_id": "tools:toolshed.g2/fastqc/0.72",
                        "details": {"id": "toolshed.g2/fastqc/0.72", "version": "0.72"},
                    },
                    {
                        "id": "toolshed.g2/fastqc/0.71",
                        "version": "0.71",
                        "resource_id": "tools:toolshed.g2/fastqc/0.71",
                        "details": {"id": "toolshed.g2/fastqc/0.71", "version": "0.71"},
                    },
                ],
            },
        ):
            response = search_tools_fn(term="FastQC")

        assert response["source"] == "tools"
        assert response["total"] == len(response["matches"])
        assert response["matches"], "Expected at least one match"

        first = response["matches"][0]
        details = first["details"]
        assert details["tool_id"] == "toolshed.g2/fastqc"
        assert details["citations"] == [
            {"title": "FastQC citation"},
            {"title": "Older citation"},
        ]
        assert len(details["versions"]) == 2

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
                search_tools_fn(term="abc")

            with pytest.raises(Exception):
                run_tool_fn("history_1", "tool1", {})
