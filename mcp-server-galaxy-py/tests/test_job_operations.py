"""Tests for job operations"""

import pytest
import responses

from galaxy_mcp.server import galaxy_state
from tests.test_helpers import get_job_details_fn


class TestJobOperations:
    def setup_method(self):
        """Set up test environment before each test"""
        galaxy_state["connected"] = True
        galaxy_state["gi"] = type("MockGI", (), {})()
        galaxy_state["url"] = "http://localhost:8080/"
        galaxy_state["api_key"] = "test_key"

    def teardown_method(self):
        """Clean up after each test"""
        galaxy_state["connected"] = False
        galaxy_state["gi"] = None

    @responses.activate
    def test_get_job_details_with_provenance(self):
        """Test getting job details using dataset provenance"""
        dataset_id = "dataset123"
        history_id = "history789"
        job_id = "job456"

        # Mock the bioblend provenance call
        mock_gi = type("MockGI", (), {})()
        mock_histories = type("MockHistories", (), {})()
        mock_histories.show_dataset_provenance = lambda history_id, dataset_id: {"job_id": job_id}
        mock_gi.histories = mock_histories
        galaxy_state["gi"] = mock_gi

        # Mock the Galaxy API job details call
        responses.add(
            responses.GET,
            f"http://localhost:8080/api/jobs/{job_id}",
            json={"id": job_id, "tool_id": "test_tool", "state": "ok"},
            status=200,
        )

        result = get_job_details_fn(dataset_id, history_id=history_id)

        assert result.success is True
        assert result.data["job"]["id"] == job_id
        assert result.data["dataset_id"] == dataset_id
        assert result.data["job_id"] == job_id

    @responses.activate
    def test_get_job_details_fallback_to_dataset_details(self):
        """Test fallback to dataset details when provenance fails"""
        dataset_id = "dataset123"
        job_id = "job456"

        # Mock the bioblend calls
        mock_gi = type("MockGI", (), {})()
        mock_histories = type("MockHistories", (), {})()
        mock_datasets = type("MockDatasets", (), {})()

        # Provenance fails
        mock_histories.show_dataset_provenance = lambda history_id, dataset_id: None
        # Dataset details has creating_job
        mock_datasets.show_dataset = lambda dataset_id: {"creating_job": job_id}

        mock_gi.histories = mock_histories
        mock_gi.datasets = mock_datasets
        galaxy_state["gi"] = mock_gi

        # Mock the Galaxy API job details call
        responses.add(
            responses.GET,
            f"http://localhost:8080/api/jobs/{job_id}",
            json={"id": job_id, "tool_id": "test_tool", "state": "ok"},
            status=200,
        )

        result = get_job_details_fn(dataset_id)

        assert result.success is True
        assert result.data["job"]["id"] == job_id
        assert result.data["dataset_id"] == dataset_id
        assert result.data["job_id"] == job_id

    def test_get_job_details_no_job_found(self):
        """Test error when no job information is found"""
        dataset_id = "dataset123"

        # Mock the bioblend calls to return no job info
        mock_gi = type("MockGI", (), {})()
        mock_histories = type("MockHistories", (), {})()
        mock_datasets = type("MockDatasets", (), {})()

        mock_histories.show_dataset_provenance = lambda history_id, dataset_id: {}
        mock_datasets.show_dataset = lambda dataset_id: {}

        mock_gi.histories = mock_histories
        mock_gi.datasets = mock_datasets
        galaxy_state["gi"] = mock_gi

        with pytest.raises(ValueError, match="No job information found"):
            get_job_details_fn(dataset_id)

    def test_get_job_details_not_connected(self):
        """Test error when not connected to Galaxy"""
        galaxy_state["connected"] = False

        with pytest.raises(ValueError, match="Not connected to Galaxy"):
            get_job_details_fn("dataset123")
