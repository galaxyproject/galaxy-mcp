"""
Test workflow-related operations
"""

from unittest.mock import Mock, patch

import pytest

from .test_helpers import (
    galaxy_state,
    get_invocations_fn,
    get_iwc_workflows_fn,
    import_workflow_from_iwc_fn,
    search_iwc_workflows_fn,
)


class TestWorkflowOperations:
    """Test workflow operations"""

    def test_get_iwc_workflows_fn(self):
        """Test getting IWC workflows"""
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "workflows": [
                    {"trs_id": "workflow1", "definition": {"name": "Test Workflow 1"}},
                    {"trs_id": "workflow2", "definition": {"name": "Test Workflow 2"}},
                ]
            }
        ]
        mock_response.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_response):
            result = get_iwc_workflows_fn()

            assert "workflows" in result
            assert len(result["workflows"]) == 2
            assert result["workflows"][0]["trs_id"] == "workflow1"

    def test_search_iwc_workflows_fn(self):
        """Test searching IWC workflows"""
        # Mock the get_iwc_workflows function since search calls it
        mock_workflows = {
            "workflows": [
                {
                    "definition": {
                        "name": "RNA-seq Analysis",
                        "annotation": "Analysis pipeline for RNA sequencing",
                        "tags": ["rna", "transcriptomics"],
                    }
                },
                {
                    "definition": {
                        "name": "DNA Variant Calling",
                        "annotation": "Pipeline for calling variants from DNA sequencing",
                        "tags": ["dna", "variants"],
                    }
                },
            ]
        }

        with patch("galaxy_mcp.server.get_iwc_workflows.fn", return_value=mock_workflows):
            result = search_iwc_workflows_fn("rna")

            assert "workflows" in result
            assert "count" in result
            assert result["count"] == 1
            assert "RNA-seq" in result["workflows"][0]["definition"]["name"]

    def test_import_workflow_from_iwc_fn(self, mock_galaxy_instance):
        """Test importing workflow from IWC"""
        # Mock the IWC API response
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "workflows": [
                    {"trsID": "test-workflow", "definition": {"name": "Test Workflow", "steps": []}}
                ]
            }
        ]
        mock_response.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_response):
            with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
                mock_galaxy_instance.workflows.import_workflow_dict.return_value = {
                    "id": "imported_workflow_1",
                    "name": "Test Workflow",
                }

                result = import_workflow_from_iwc_fn("test-workflow")

                assert result["imported_workflow"]["id"] == "imported_workflow_1"
                assert result["imported_workflow"]["name"] == "Test Workflow"

    def test_get_invocations_fn(self, mock_galaxy_instance):
        """Test getting workflow invocations"""
        mock_galaxy_instance.invocations.get_invocations.return_value = [
            {"id": "invocation_1", "state": "scheduled"},
            {"id": "invocation_2", "state": "running"},
        ]

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = get_invocations_fn()

            assert isinstance(result, dict)
            assert "invocations" in result
            assert len(result["invocations"]) == 2
            assert result["invocations"][0]["id"] == "invocation_1"

    def test_workflow_operations_not_connected(self):
        """Test workflow operations fail when not connected"""
        with patch.dict(galaxy_state, {"connected": False}):
            with pytest.raises(Exception):
                get_invocations_fn()

            # IWC operations don't require connection
            # But import_workflow_from_iwc does require connection
            with pytest.raises(Exception):
                import_workflow_from_iwc_fn("test-workflow")
