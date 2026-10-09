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


class TestGetJobDetailsByJobId:
    """The job asked for by its own id, which skips both lookups and reads the job directly."""

    BASE = "http://localhost:8080"
    JOB_ID = "j0000001"

    def setup_method(self):
        galaxy_state["connected"] = True
        galaxy_state["gi"] = type("MockGI", (), {})()
        galaxy_state["url"] = f"{self.BASE}/"
        galaxy_state["api_key"] = "test_key"

    def teardown_method(self):
        galaxy_state["connected"] = False
        galaxy_state["gi"] = None

    @responses.activate
    def test_by_job_id_reads_the_job_and_names_no_dataset(self):
        responses.add(
            responses.GET,
            f"{self.BASE}/api/jobs/{self.JOB_ID}",
            json={"id": self.JOB_ID, "tool_id": "fastqc", "state": "ok"},
            status=200,
        )

        result = get_job_details_fn(job_id=self.JOB_ID)

        assert result.success is True
        assert result.data == {
            "job": {"id": self.JOB_ID, "tool_id": "fastqc", "state": "ok"},
            "dataset_id": None,
            "job_id": self.JOB_ID,
        }
        assert result.message == f"Retrieved job details for job '{self.JOB_ID}'"
        assert "full" not in responses.calls[0].request.url

    @responses.activate
    def test_full_sends_the_flag_to_galaxy(self):
        responses.add(
            responses.GET,
            f"{self.BASE}/api/jobs/{self.JOB_ID}",
            json={"id": self.JOB_ID, "params": {}, "inputs": {}, "outputs": {}},
            status=200,
            match=[responses.matchers.query_param_matcher({"full": "true"})],
        )

        result = get_job_details_fn(job_id=self.JOB_ID, full=True)

        assert result.success is True
        assert result.data["job"]["params"] == {}

    @responses.activate
    def test_full_also_applies_when_the_job_is_found_through_a_dataset(self):
        mock_gi = type("MockGI", (), {})()
        mock_datasets = type("MockDatasets", (), {})()
        mock_datasets.show_dataset = lambda dataset_id: {"creating_job": self.JOB_ID}
        mock_gi.datasets = mock_datasets
        galaxy_state["gi"] = mock_gi
        responses.add(
            responses.GET,
            f"{self.BASE}/api/jobs/{self.JOB_ID}",
            json={"id": self.JOB_ID},
            status=200,
            match=[responses.matchers.query_param_matcher({"full": "true"})],
        )

        result = get_job_details_fn("d0001", full=True)

        assert result.data["dataset_id"] == "d0001"
        assert result.message == "Retrieved job details for dataset 'd0001'"

    @pytest.mark.parametrize("status", [400, 404])
    @responses.activate
    def test_an_unknown_or_malformed_id_is_not_found(self, status):
        responses.add(
            responses.GET,
            f"{self.BASE}/api/jobs/{self.JOB_ID}",
            json={"err_msg": "no such job", "err_code": status * 1000},
            status=status,
        )

        with pytest.raises(ValueError, match=f"Job ID '{self.JOB_ID}' not found or not accessible"):
            get_job_details_fn(job_id=self.JOB_ID)

    @responses.activate
    def test_any_other_status_goes_through_format_error(self):
        responses.add(responses.GET, f"{self.BASE}/api/jobs/{self.JOB_ID}", status=500)

        with pytest.raises(ValueError, match="Get job details failed: 500") as raised:
            get_job_details_fn(job_id=self.JOB_ID)

        assert str(raised.value).endswith(f"Context: job_id={self.JOB_ID}")

    def test_both_ids_are_refused_before_any_request(self):
        with pytest.raises(ValueError, match="not both"):
            get_job_details_fn("d0001", job_id=self.JOB_ID)

    def test_neither_id_is_refused_before_any_request(self):
        with pytest.raises(ValueError, match="neither was given"):
            get_job_details_fn()
