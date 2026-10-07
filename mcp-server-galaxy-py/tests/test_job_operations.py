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


class TestJobLogs:
    """get_job_details reads the job in full and keeps both ends of a long log."""

    def setup_method(self):
        mock_gi = type("MockGI", (), {})()
        mock_datasets = type("MockDatasets", (), {})()
        mock_datasets.show_dataset = lambda dataset_id: {"id": dataset_id, "creating_job": "j1"}
        mock_gi.datasets = mock_datasets
        galaxy_state["connected"] = True
        galaxy_state["gi"] = mock_gi
        galaxy_state["url"] = "http://localhost:8080/"
        galaxy_state["api_key"] = "test_key"

    def teardown_method(self):
        galaxy_state["connected"] = False
        galaxy_state["gi"] = None

    def _job(self, **fields):
        responses.add(
            responses.GET,
            "http://localhost:8080/api/jobs/j1",
            json={"id": "j1", "state": "error", **fields},
            match=[responses.matchers.query_param_matcher({"full": "true"})],
        )
        return get_job_details_fn("d1").data["job"]

    @responses.activate
    def test_asks_for_the_job_in_full(self):
        assert self._job(tool_stderr="Traceback: boom")["tool_stderr"] == "Traceback: boom"

    @responses.activate
    def test_keeps_the_first_line_through_a_flood_of_warnings(self):
        noise = "\n".join(["Invalid bed line (skipped): @SQ SN:chr1 LN:248956422"] * 600)
        out = self._job(tool_stderr="Reading reference bed file: ref.dat\n" + noise)["tool_stderr"]
        assert out.startswith("Reading reference bed file: ref.dat")
        assert "bytes omitted" in out

    @responses.activate
    def test_keeps_the_end_and_whole_lines_at_both_cuts(self):
        noisy = "\n".join(f"warning number {i}" for i in range(2000))
        out = self._job(tool_stderr=noisy + "\nRuntimeError: the real cause")["tool_stderr"]
        lines = out.split("\n")
        assert lines[0] == "warning number 0"
        assert lines[-1] == "RuntimeError: the real cause"
        total = len((noisy + "\nRuntimeError: the real cause").encode())
        marker = next(line for line in lines if "omitted" in line)
        assert marker.endswith(f"of {total} bytes omitted ...]")
        assert len(out.encode()) <= 4096 + len("\n[... 00000 of 000000 bytes omitted ...]\n")

    @responses.activate
    def test_trims_every_log_field_and_leaves_the_rest_of_the_job_alone(self):
        long = "\n".join(f"line {i}" for i in range(3000))
        logs = dict.fromkeys(("tool_stdout", "job_stderr", "stdout"), long)
        job = self._job(**logs, params={"input": "d0"})
        assert all("bytes omitted" in job[f] for f in ("tool_stdout", "job_stderr", "stdout"))
        assert job["params"] == {"input": "d0"}

    @responses.activate
    def test_measures_in_utf8_bytes(self):
        # 3 bytes per character: 1500 of them is over 4 KB though it is not 4096 characters.
        out = self._job(tool_stderr="\n".join(["€" * 50] * 30))["tool_stderr"]
        assert "bytes omitted" in out
