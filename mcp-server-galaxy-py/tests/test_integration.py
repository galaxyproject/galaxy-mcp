"""
Integration tests for complete workflows
"""

from unittest.mock import Mock, patch

import pytest

from .test_helpers import (
    create_history_fn,
    galaxy_state,
    import_workflow_from_iwc_fn,
    list_history_ids_fn,
    run_tool_fn,
    upload_file_fn,
)


class TestIntegration:
    """Integration tests for complete user workflows"""

    def test_complete_analysis_workflow(self, mock_galaxy_instance):
        """Test a complete analysis workflow from upload to results"""
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            # 1. Create a new history
            mock_galaxy_instance.histories.create_history.return_value = {
                "id": "new_history_1",
                "name": "Analysis History",
            }

            history = create_history_fn("Analysis History")
            assert history["id"] == "new_history_1"

            # 2. Upload a file
            mock_galaxy_instance.tools.upload_file.return_value = {
                "outputs": [{"id": "uploaded_dataset_1", "name": "input.fasta"}]
            }

            with patch("os.path.exists", return_value=True):
                dataset = upload_file_fn("/path/to/input.fasta", history["id"])
                assert dataset["outputs"][0]["id"] == "uploaded_dataset_1"

            # 3. Run a tool on the uploaded file

            mock_galaxy_instance.tools.run_tool.return_value = {
                "jobs": [{"id": "job_1", "state": "ok"}],
                "outputs": [{"id": "output_dataset_1", "name": "aligned.bam"}],
            }

            tool_result = run_tool_fn(
                history["id"],
                "bwa",
                {"input": {"src": "hda", "id": dataset["outputs"][0]["id"]}, "reference": "hg38"},
            )
            assert tool_result["outputs"][0]["id"] == "output_dataset_1"

            # 4. Download functionality not implemented yet
            # Future: Implement dataset download functionality

    def test_workflow_execution_pipeline(self, mock_galaxy_instance):
        """Test running a complete workflow pipeline"""
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            # 1. Import a workflow

            mock_response = Mock()
            mock_response.json.return_value = {
                "name": "RNA-seq Pipeline",
                "steps": [{"id": 0, "tool_id": "fastqc"}],
            }
            mock_response.raise_for_status.return_value = None

            # Mock the IWC API
            mock_iwc_response = Mock()
            mock_iwc_response.json.return_value = [
                {
                    "workflows": [
                        {
                            "trsID": "workflows/rnaseq-pe",
                            "definition": {"name": "RNA-seq Pipeline", "steps": []},
                        }
                    ]
                }
            ]
            mock_iwc_response.raise_for_status.return_value = None

            with patch("requests.get", return_value=mock_iwc_response):
                mock_galaxy_instance.workflows.import_workflow_dict.return_value = {
                    "id": "imported_workflow_1",
                    "name": "RNA-seq Pipeline",
                }

                result = import_workflow_from_iwc_fn("workflows/rnaseq-pe")
                assert result["imported_workflow"]["id"] == "imported_workflow_1"

            # 2. Prepare input datasets

            mock_galaxy_instance.histories.get_histories.return_value = [
                {"id": "history_1", "name": "RNA-seq Data"}
            ]

            histories = list_history_ids_fn()
            histories[0]["id"]

            # 3. Workflow execution not directly available in current MCP implementation
            # Future: Add workflow invocation functionality

    def test_error_handling_cascade(self, mock_galaxy_instance):
        """Test proper error handling throughout the stack"""
        with patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance}):
            # Test error in tool execution cascades properly

            mock_galaxy_instance.tools.run_tool.side_effect = Exception("Tool not found")

            with pytest.raises(ValueError, match="Run tool failed"):
                run_tool_fn("history_1", "nonexistent_tool", {})

            # Test error in workflow import

            with patch("requests.get", side_effect=Exception("Network error")):
                with pytest.raises(ValueError, match="Failed to import workflow from IWC"):
                    import_workflow_from_iwc_fn("nonexistent-workflow")
