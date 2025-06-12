"""
Test tool-related operations
"""

from unittest.mock import patch

import pytest

from .test_helpers import galaxy_state, run_tool_fn, search_tools_fn


class TestToolOperations:
    """Test tool operations"""

    def test_search_tools_fn(self, mock_galaxy_instance):
        """Test tool search functionality"""
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            # Search should return dict with 'tools' key
            result = search_tools_fn("")
            assert "tools" in result
            assert len(result["tools"]) == 2
            assert result["tools"][0]["id"] == "tool1"

            # Search with query
            mock_galaxy_instance.tools.get_tools.return_value = [
                {"id": "tool1", "name": "Test Tool 1", "description": "Aligns sequences"}
            ]

            result = search_tools_fn("align")
            assert "tools" in result
            assert len(result["tools"]) == 1
            assert "align" in result["tools"][0]["description"].lower()

    def test_search_tools_with_results(self, mock_galaxy_instance):
        """Test search tools returns filtered results"""
        all_tools = [
            {"id": "tool1", "name": "BWA Aligner", "description": "Aligns sequences"},
            {"id": "tool2", "name": "Samtools", "description": "Process BAM files"},
            {"id": "tool3", "name": "HISAT2", "description": "Fast aligner"},
        ]

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            # Mock filtering behavior
            def mock_get_tools(name=None):
                if name and name.lower() == "align":
                    return [
                        t
                        for t in all_tools
                        if "align" in t["name"].lower() or "align" in t["description"].lower()
                    ]
                return all_tools

            mock_galaxy_instance.tools.get_tools.side_effect = mock_get_tools

            # Search for aligners
            result = search_tools_fn("align")
            assert "tools" in result
            aligners = result["tools"]
            assert len(aligners) == 2
            assert any("BWA" in t["name"] for t in aligners)
            assert any("HISAT2" in t["name"] for t in aligners)

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
                search_tools_fn("query")

            with pytest.raises(Exception):
                run_tool_fn("history_1", "tool1", {})
