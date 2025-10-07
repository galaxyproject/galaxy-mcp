"""
Test workflow-related operations
"""

from unittest.mock import Mock, patch

import pytest

from .test_helpers import (
    cancel_workflow_invocation_fn,
    galaxy_state,
    get_invocations_fn,
    get_iwc_workflows_fn,
    get_workflow_details_fn,
    import_workflow_from_iwc_fn,
    invoke_workflow_fn,
    list_workflows_fn,
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

            # New workflow operations should fail when not connected
            with pytest.raises(Exception):
                list_workflows_fn()

            with pytest.raises(Exception):
                get_workflow_details_fn("test-workflow-id")

            with pytest.raises(Exception):
                invoke_workflow_fn("test-workflow-id")

            with pytest.raises(Exception):
                cancel_workflow_invocation_fn("test-invocation-id")

    def test_list_workflows_fn(self, mock_galaxy_instance):
        """Test listing workflows"""
        mock_workflows = [
            {
                "id": "workflow1",
                "name": "Test Workflow 1",
                "published": False,
                "owner": "test_user",
                "version": 1,
            },
            {
                "id": "workflow2",
                "name": "RNA-seq Analysis",
                "published": True,
                "owner": "admin_user",
                "version": 2,
            },
        ]

        mock_galaxy_instance.workflows.get_workflows.return_value = mock_workflows

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            # Test getting all workflows
            result = list_workflows_fn()

            assert "workflows" in result
            assert len(result["workflows"]) == 2
            assert result["workflows"][0]["id"] == "workflow1"
            assert result["workflows"][1]["name"] == "RNA-seq Analysis"

            # Verify function was called with correct parameters
            mock_galaxy_instance.workflows.get_workflows.assert_called_with(
                workflow_id=None, name=None, published=False
            )

    def test_list_workflows_fn_with_filters(self, mock_galaxy_instance):
        """Test listing workflows with filters"""
        mock_workflows = [
            {
                "id": "workflow1",
                "name": "RNA-seq Analysis",
                "published": True,
                "owner": "admin_user",
            }
        ]

        mock_galaxy_instance.workflows.get_workflows.return_value = mock_workflows

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = list_workflows_fn(name="RNA-seq", published=True)

            assert "workflows" in result
            assert len(result["workflows"]) == 1

            # Verify function was called with filters
            mock_galaxy_instance.workflows.get_workflows.assert_called_with(
                workflow_id=None, name="RNA-seq", published=True
            )

    def test_get_workflow_details_fn(self, mock_galaxy_instance):
        """Test getting workflow details"""
        mock_workflow = {
            "id": "workflow1",
            "name": "Test Workflow",
            "version": 1,
            "steps": {
                "0": {"tool_id": "upload1", "type": "data_input", "annotation": "Input file"},
                "1": {"tool_id": "fastqc", "type": "tool", "annotation": "Quality control"},
            },
            "inputs": {"0": {"label": "Input Dataset", "uuid": "abc123"}},
        }

        mock_galaxy_instance.workflows.show_workflow.return_value = mock_workflow

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = get_workflow_details_fn("workflow1")

            assert "workflow" in result
            assert result["workflow"]["id"] == "workflow1"
            assert result["workflow"]["name"] == "Test Workflow"
            assert "steps" in result["workflow"]
            assert "inputs" in result["workflow"]

            # Verify function was called correctly
            mock_galaxy_instance.workflows.show_workflow.assert_called_with(
                workflow_id="workflow1", version=None
            )

    def test_get_workflow_details_fn_with_version(self, mock_galaxy_instance):
        """Test getting workflow details with specific version"""
        mock_workflow = {"id": "workflow1", "name": "Test Workflow", "version": 2}

        mock_galaxy_instance.workflows.show_workflow.return_value = mock_workflow

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = get_workflow_details_fn("workflow1", version=2)

            assert "workflow" in result
            assert result["workflow"]["version"] == 2

            # Verify version parameter was passed
            mock_galaxy_instance.workflows.show_workflow.assert_called_with(
                workflow_id="workflow1", version=2
            )

    def test_invoke_workflow_fn(self, mock_galaxy_instance):
        """Test invoking a workflow"""
        mock_invocation = {
            "id": "invocation123",
            "state": "scheduled",
            "workflow_id": "workflow1",
            "history_id": "history1",
            "steps": [],
        }

        mock_galaxy_instance.workflows.invoke_workflow.return_value = mock_invocation

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            inputs = {"0": {"id": "dataset123", "src": "hda"}}
            params = {"1": {"param1": "value1"}}

            result = invoke_workflow_fn(
                workflow_id="workflow1", inputs=inputs, params=params, history_id="history1"
            )

            assert "invocation" in result
            assert result["invocation"]["id"] == "invocation123"
            assert result["invocation"]["state"] == "scheduled"

            # Verify function was called with correct parameters
            mock_galaxy_instance.workflows.invoke_workflow.assert_called_with(
                workflow_id="workflow1",
                inputs=inputs,
                params=params,
                history_id="history1",
                history_name=None,
                inputs_by="step_index",
                parameters_normalized=False,
            )

    def test_invoke_workflow_fn_with_history_name(self, mock_galaxy_instance):
        """Test invoking workflow with new history name"""
        mock_invocation = {
            "id": "invocation456",
            "state": "scheduled",
            "workflow_id": "workflow1",
            "history_id": "new_history123",
        }

        mock_galaxy_instance.workflows.invoke_workflow.return_value = mock_invocation

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = invoke_workflow_fn(
                workflow_id="workflow1", history_name="RNA-seq Analysis Results", inputs_by="name"
            )

            assert "invocation" in result
            assert result["invocation"]["history_id"] == "new_history123"

            # Verify function was called with history_name
            mock_galaxy_instance.workflows.invoke_workflow.assert_called_with(
                workflow_id="workflow1",
                inputs=None,
                params=None,
                history_id=None,
                history_name="RNA-seq Analysis Results",
                inputs_by="name",
                parameters_normalized=False,
            )

    def test_cancel_workflow_invocation_fn(self, mock_galaxy_instance):
        """Test cancelling a workflow invocation"""
        mock_cancelled_invocation = {
            "id": "invocation123",
            "state": "cancelled",
            "workflow_id": "workflow1",
        }

        mock_galaxy_instance.workflows.cancel_invocation.return_value = mock_cancelled_invocation

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = cancel_workflow_invocation_fn("invocation123")

            assert "cancelled" in result
            assert result["cancelled"] is True
            assert "invocation" in result
            assert result["invocation"]["state"] == "cancelled"

            # Verify function was called correctly
            mock_galaxy_instance.workflows.cancel_invocation.assert_called_with("invocation123")

    def test_workflow_operations_error_handling(self, mock_galaxy_instance):
        """Test error handling in workflow operations"""
        # Test list_workflows error
        mock_galaxy_instance.workflows.get_workflows.side_effect = Exception("API Error")

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError, match="List workflows failed"):
                list_workflows_fn()

        # Test get_workflow_details error
        mock_galaxy_instance.workflows.show_workflow.side_effect = Exception("Workflow not found")

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError, match="Get workflow details failed"):
                get_workflow_details_fn("invalid_id")

        # Test invoke_workflow error
        mock_galaxy_instance.workflows.invoke_workflow.side_effect = Exception("Invalid inputs")

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError, match="Invoke workflow failed"):
                invoke_workflow_fn("workflow1")

        # Test cancel_workflow_invocation error
        mock_galaxy_instance.workflows.cancel_invocation.side_effect = Exception(
            "Invocation not found"
        )

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            with pytest.raises(ValueError, match="Cancel workflow invocation failed"):
                cancel_workflow_invocation_fn("invalid_invocation")
