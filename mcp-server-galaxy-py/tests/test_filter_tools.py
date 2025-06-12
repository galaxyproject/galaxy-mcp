"""Tests for filter_tools_by_dataset functionality"""

from unittest.mock import MagicMock, patch

import pytest

from .test_helpers import filter_tools_by_dataset_fn, galaxy_state


class TestFilterTools:
    """Test suite for filter_tools_by_dataset function"""

    @pytest.fixture()
    def mock_galaxy_instance(self):
        """Create a mock Galaxy instance"""
        mock_gi = MagicMock()
        return mock_gi

    @pytest.fixture()
    def mock_tool_panel(self):
        """Create a mock tool panel structure"""
        return [
            {
                "id": "csv_tool",
                "name": "CSV Parser",
                "description": "Parse CSV files",
                "versions": ["1.0", "1.1"],
            },
            {
                "id": "tabular_tool",
                "name": "Tabular Processor",
                "description": "Process tabular data",
                "versions": ["2.0"],
            },
            {
                "elems": [
                    {
                        "id": "fasta_tool",
                        "name": "FASTA Reader",
                        "description": "Read FASTA sequences",
                        "versions": ["1.0"],
                    },
                    {
                        "id": "generic_tool",
                        "name": "Generic Tool",
                        "description": "A generic processing tool",
                        "versions": ["3.0"],
                    },
                ]
            },
        ]

    def test_filter_tools_by_dataset_single_keyword(self, mock_galaxy_instance, mock_tool_panel):
        """Test filtering with a single dataset type keyword"""
        mock_galaxy_instance.tools.get_tool_panel.return_value = mock_tool_panel
        mock_galaxy_instance.tools.show_tool.return_value = {
            "inputs": [{"extensions": ["txt", "data"]}]
        }

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = filter_tools_by_dataset_fn(["csv"])

            assert "recommended_tools" in result
            assert "count" in result
            assert result["count"] >= 1

            # Check that CSV tool is in results
            tool_names = [tool["name"] for tool in result["recommended_tools"]]
            assert "CSV Parser" in tool_names

    def test_filter_tools_by_dataset_multiple_keywords(self, mock_galaxy_instance, mock_tool_panel):
        """Test filtering with multiple dataset type keywords"""
        mock_galaxy_instance.tools.get_tool_panel.return_value = mock_tool_panel
        mock_galaxy_instance.tools.show_tool.return_value = {
            "inputs": [{"extensions": ["txt", "data"]}]
        }

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = filter_tools_by_dataset_fn(["csv", "tabular"])

            assert "recommended_tools" in result
            assert "count" in result
            assert result["count"] >= 2

            # Check that both CSV and tabular tools are in results
            tool_names = [tool["name"] for tool in result["recommended_tools"]]
            assert "CSV Parser" in tool_names
            assert "Tabular Processor" in tool_names

    def test_filter_tools_by_input_extensions(self, mock_galaxy_instance, mock_tool_panel):
        """Test filtering tools by their input extensions"""
        mock_galaxy_instance.tools.get_tool_panel.return_value = mock_tool_panel

        # Mock different extensions for different tools
        def mock_show_tool(tool_id, io_details=False):
            if tool_id == "generic_tool":
                return {"inputs": [{"extensions": ["csv", "tsv"]}]}
            return {"inputs": [{"extensions": ["txt"]}]}

        mock_galaxy_instance.tools.show_tool.side_effect = mock_show_tool

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = filter_tools_by_dataset_fn(["csv"])

            # Generic tool should be included due to csv extension
            tool_ids = [tool["id"] for tool in result["recommended_tools"]]
            assert "csv_tool" in tool_ids  # matches by name
            assert "generic_tool" in tool_ids  # matches by extension

    def test_filter_tools_not_connected(self):
        """Test that filtering fails when not connected to Galaxy"""
        with patch.dict(galaxy_state, {"connected": False}):
            with pytest.raises(ValueError, match="Not connected to Galaxy"):
                filter_tools_by_dataset_fn(["csv"])

    def test_filter_tools_handles_tool_panel_error(self, mock_galaxy_instance):
        """Test error handling when tool panel retrieval fails"""
        mock_galaxy_instance.tools.get_tool_panel.side_effect = Exception("API Error")

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError, match="Failed to filter tools based on dataset"):
                filter_tools_by_dataset_fn(["csv"])

    def test_filter_tools_skips_label_tools(self, mock_galaxy_instance):
        """Test that tools with IDs ending in _label are skipped"""
        mock_tool_panel = [
            {
                "id": "section_label",
                "name": "Section Label",
                "description": "This is not a real tool",
            },
            {"id": "real_tool", "name": "Real Tool", "description": "Process CSV files"},
        ]

        mock_galaxy_instance.tools.get_tool_panel.return_value = mock_tool_panel
        mock_galaxy_instance.tools.show_tool.return_value = {"inputs": [{"extensions": ["csv"]}]}

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = filter_tools_by_dataset_fn(["csv"])

            # Only real_tool should be in results
            tool_ids = [tool["id"] for tool in result["recommended_tools"]]
            assert "section_label" not in tool_ids
            assert "real_tool" in tool_ids

    def test_filter_tools_handles_show_tool_error(self, mock_galaxy_instance):
        """Test that individual tool errors are handled gracefully"""
        # Create a tool panel with tools that don't match by name/description
        mock_tool_panel = [
            {"id": "tool1", "name": "Tool One", "description": "First tool"},
            {"id": "tool2", "name": "Tool Two", "description": "Second tool"},
        ]
        mock_galaxy_instance.tools.get_tool_panel.return_value = mock_tool_panel

        # Make show_tool fail for tool2
        def mock_show_tool(tool_id, io_details=False):
            if tool_id == "tool2":
                raise Exception("Tool not found")
            # tool1 will have csv in extensions
            return {"inputs": [{"extensions": ["csv"]}]}

        mock_galaxy_instance.tools.show_tool.side_effect = mock_show_tool

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            # Should not raise, just skip the failing tool
            result = filter_tools_by_dataset_fn(["csv"])

            # Only tool1 should be in results
            tool_ids = [tool["id"] for tool in result["recommended_tools"]]
            assert "tool1" in tool_ids
            assert "tool2" not in tool_ids
