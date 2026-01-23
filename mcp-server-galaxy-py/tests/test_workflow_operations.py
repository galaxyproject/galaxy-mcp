"""
Test workflow-related operations
"""

from unittest.mock import patch

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
        mock_manifest = [
            {
                "workflows": [
                    {"trs_id": "workflow1", "definition": {"name": "Test Workflow 1"}},
                    {"trs_id": "workflow2", "definition": {"name": "Test Workflow 2"}},
                ]
            }
        ]

        with patch("galaxy_mcp.server.get_manifest_json", return_value=mock_manifest):
            result = get_iwc_workflows_fn()

            assert result.success is True
            assert result.count == 2
            assert len(result.data) == 2
            assert result.data[0]["trs_id"] == "workflow1"

    def test_search_iwc_workflows_fn(self):
        """Test searching IWC workflows"""
        # Mock the manifest data that get_manifest_json returns
        mock_manifest = [
            {
                "workflows": [
                    {
                        "trsID": "workflow-rna-seq",
                        "definition": {
                            "name": "RNA-seq Analysis",
                            "annotation": "Analysis pipeline for RNA sequencing",
                            "tags": ["rna", "transcriptomics"],
                        },
                    },
                    {
                        "trsID": "workflow-dna-variant",
                        "definition": {
                            "name": "DNA Variant Calling",
                            "annotation": "Pipeline for calling variants from DNA sequencing",
                            "tags": ["dna", "variants"],
                        },
                    },
                ]
            }
        ]

        with patch("galaxy_mcp.server.get_manifest_json", return_value=mock_manifest):
            result = search_iwc_workflows_fn("rna")

            assert result.success is True
            assert result.count == 1
            # New API returns simplified structure with name at top level
            assert "RNA-seq" in result.data[0]["name"]
            assert result.data[0]["trsID"] == "workflow-rna-seq"

    def test_import_workflow_from_iwc_fn(self, mock_galaxy_instance):
        """Test importing workflow from IWC"""
        # Mock the manifest data that get_manifest_json returns
        mock_manifest = [
            {
                "workflows": [
                    {"trsID": "test-workflow", "definition": {"name": "Test Workflow", "steps": []}}
                ]
            }
        ]

        with patch("galaxy_mcp.server.get_manifest_json", return_value=mock_manifest):
            with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
                mock_galaxy_instance.workflows.import_workflow_dict.return_value = {
                    "id": "imported_workflow_1",
                    "name": "Test Workflow",
                }

                result = import_workflow_from_iwc_fn("test-workflow")

                assert result.success is True
                assert result.data["id"] == "imported_workflow_1"
                assert result.data["name"] == "Test Workflow"

    def test_get_invocations_fn(self, mock_galaxy_instance):
        """Test getting workflow invocations"""
        mock_galaxy_instance.invocations.get_invocations.return_value = [
            {"id": "invocation_1", "state": "scheduled"},
            {"id": "invocation_2", "state": "running"},
        ]

        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            result = get_invocations_fn()

            assert result.success is True
            assert result.count == 2
            assert len(result.data) == 2
            assert result.data[0]["id"] == "invocation_1"

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

            assert result.success is True
            assert result.count == 2
            assert len(result.data) == 2
            assert result.data[0]["id"] == "workflow1"
            assert result.data[1]["name"] == "RNA-seq Analysis"

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

            assert result.success is True
            assert result.count == 1
            assert len(result.data) == 1

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

            assert result.success is True
            assert result.data["id"] == "workflow1"
            assert result.data["name"] == "Test Workflow"
            assert "steps" in result.data
            assert "inputs" in result.data

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

            assert result.success is True
            assert result.data["version"] == 2

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

            assert result.success is True
            assert result.data["id"] == "invocation123"
            assert result.data["state"] == "scheduled"

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

            assert result.success is True
            assert result.data["history_id"] == "new_history123"

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

            assert result.success is True
            assert result.data["cancelled"] is True
            assert result.data["invocation"]["state"] == "cancelled"

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
