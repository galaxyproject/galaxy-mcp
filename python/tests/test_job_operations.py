"""Tests for job operations"""

import re
from unittest.mock import Mock, patch

import bioblend
import pytest
import responses

from galaxy_mcp.server import galaxy_state
from tests.test_helpers import get_job_details_fn, list_jobs_fn


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
    JOB_ID = "0123456789abcdef"

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
        # What full adds on top of the plain read, which already carries params/inputs/outputs.
        full_job = {
            "id": self.JOB_ID,
            "params": {},
            "inputs": {},
            "outputs": {},
            "job_stdout": "done",
            "job_stderr": "",
            "job_messages": [],
            "dependencies": [],
            "job_metrics": [],
        }
        responses.add(
            responses.GET,
            f"{self.BASE}/api/jobs/{self.JOB_ID}",
            json=full_job,
            status=200,
            match=[responses.matchers.query_param_matcher({"full": "true"})],
        )

        result = get_job_details_fn(job_id=self.JOB_ID, full=True)

        assert result.success is True
        assert result.data["job"] == full_job

    def test_full_is_described_by_what_it_adds(self):
        # The non-full job already carries params, inputs and outputs (EncodedJobDetails in
        # Galaxy's schema requires all three), so the description must not send an agent to
        # full=true for them; it names the extra, heavier fields the flag actually brings.
        from tests.surface_manifest import build_manifest

        tools = {tool["name"]: tool for tool in build_manifest()["tools"]}
        served = tools["get_job_details"]["inputSchema"]["properties"]["full"]["description"]
        description = " ".join(served.split())  # the docstring's line breaks travel with it
        for added in ("stdout", "stderr", "job messages", "dependencies", "job metrics"):
            assert added in description
        assert "Params, inputs and outputs are in the plain read already" in description

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

    @responses.activate
    def test_the_old_positional_pair_is_still_dataset_then_history(self):
        """get_job_details(dataset_id, history_id) predates job_id, so job_id sits after both.

        Ordered the other way, a caller's positional history id would land in job_id and be
        refused as a second id rather than used for the provenance lookup.
        """
        asked: list[tuple[str, str]] = []
        mock_gi = type("MockGI", (), {})()
        mock_histories = type("MockHistories", (), {})()
        mock_histories.show_dataset_provenance = lambda history_id, dataset_id: (
            asked.append((history_id, dataset_id)) or {"job_id": self.JOB_ID}
        )
        mock_gi.histories = mock_histories
        galaxy_state["gi"] = mock_gi
        responses.add(
            responses.GET, f"{self.BASE}/api/jobs/{self.JOB_ID}", json={"id": self.JOB_ID}
        )

        result = get_job_details_fn("d0001", "h0001")

        assert asked == [("h0001", "d0001")]
        assert result.data["dataset_id"] == "d0001"
        assert result.data["job_id"] == self.JOB_ID

    @pytest.mark.parametrize("value", [".", "../histories", "not-an-id", "j0000001"])
    @responses.activate
    def test_a_value_that_is_not_an_id_is_refused_before_anything_is_sent(self, value):
        """An id travels as one path segment, and requests folds "." and ".." into the path.

        Galaxy would answer "not-an-id" with a 400 anyway; "." would never reach it as an
        id at all, but as GET /api/jobs/, which is the listing.
        """
        with pytest.raises(ValueError, match="not a Galaxy id") as exc:
            get_job_details_fn(job_id=value)

        assert str(exc.value).endswith("Nothing was sent to Galaxy.")
        assert len(responses.calls) == 0

    def test_both_ids_are_refused_before_the_shape_of_either_is_looked_at(self):
        with pytest.raises(ValueError, match="not both"):
            get_job_details_fn("d0001", job_id=".")

    @pytest.mark.parametrize(
        "answer",
        [{"id": "fedcba9876543210", "state": "ok"}, [{"id": "0123456789abcdef"}], {"state": "ok"}],
    )
    @responses.activate
    def test_a_200_that_is_not_this_jobs_record_is_refused(self, answer):
        responses.add(responses.GET, f"{self.BASE}/api/jobs/{self.JOB_ID}", json=answer)

        with pytest.raises(ValueError, match="not that job's record"):
            get_job_details_fn(job_id=self.JOB_ID)

    @responses.activate
    def test_the_ids_case_is_not_held_against_the_caller(self):
        """Galaxy decodes either case and writes lowercase, so the two spell one record."""
        responses.add(
            responses.GET, f"{self.BASE}/api/jobs/{self.JOB_ID.upper()}", json={"id": self.JOB_ID}
        )

        result = get_job_details_fn(job_id=self.JOB_ID.upper())

        assert result.success is True


class TestListJobs:
    """list_jobs: one GET on the job index, filters sent only when they say something."""

    @staticmethod
    def _connected(mock_galaxy_instance):
        # Mock(spec=GalaxyInstance) knows the class, and a real instance only grows a
        # jobs client in __init__, so the attribute is put there by hand.
        mock_galaxy_instance.jobs = Mock()
        return patch.dict(galaxy_state, {"connected": True, "gi": mock_galaxy_instance})

    def test_sends_the_window_and_every_filter_that_is_set(self, mock_galaxy_instance):
        rows = [
            {"id": "job1", "state": "ok", "tool_id": "cat1"},
            {"id": "job2", "state": "ok", "tool_id": "cat1"},
        ]
        with self._connected(mock_galaxy_instance):
            mock_galaxy_instance.jobs._get.return_value = rows
            result = list_jobs_fn(
                history_id="h1",
                state="ok",
                date_range_min="2026-01-01T00:00:00",
                date_range_max="2026-02-01",
                order_by="create_time",
                limit=5,
                offset=10,
            )

        mock_galaxy_instance.jobs._get.assert_called_once_with(
            params={
                "limit": 5,
                "offset": 10,
                "history_id": "h1",
                "state": "ok",
                "date_range_min": "2026-01-01T00:00:00",
                "date_range_max": "2026-02-01",
                "order_by": "create_time",
                "view": "collection",
            }
        )
        assert result.success is True
        assert result.data == rows
        assert result.count == 2
        assert result.message == "Retrieved 2 jobs"
        # Galaxy reports no total for this index, so there is no window to describe.
        assert result.pagination is None

    def test_a_blank_filter_is_not_sent(self, mock_galaxy_instance):
        with self._connected(mock_galaxy_instance):
            mock_galaxy_instance.jobs._get.return_value = []
            result = list_jobs_fn(history_id="", state=None, date_range_min="")

        mock_galaxy_instance.jobs._get.assert_called_once_with(
            params={"limit": 100, "offset": 0, "order_by": "update_time", "view": "collection"}
        )
        assert result.data == []
        assert result.count == 0
        assert result.message == "Retrieved 0 jobs"

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"limit": 0}, "limit must be at least 1 (got 0)"),
            ({"offset": -1}, "offset must be 0 or greater (got -1)"),
        ],
    )
    def test_refuses_a_bad_window_before_asking_galaxy(self, mock_galaxy_instance, kwargs, message):
        with self._connected(mock_galaxy_instance):
            with pytest.raises(ValueError, match=re.escape(message)):
                list_jobs_fn(**kwargs)
        mock_galaxy_instance.jobs._get.assert_not_called()

    def test_has_no_upper_cap_on_limit(self, mock_galaxy_instance):
        # Galaxy's job index takes any limit, and so does this tool; the sentence the
        # capped listings refuse with must not appear here.
        with self._connected(mock_galaxy_instance):
            mock_galaxy_instance.jobs._get.return_value = []
            list_jobs_fn(limit=5000)
        assert mock_galaxy_instance.jobs._get.call_args.kwargs["params"]["limit"] == 5000

    def test_refuses_an_error_body_under_a_200(self, mock_galaxy_instance):
        with self._connected(mock_galaxy_instance):
            mock_galaxy_instance.jobs._get.return_value = {
                "err_msg": "History is not accessible by user",
                "err_code": 403002,
            }
            with pytest.raises(ValueError, match="List jobs failed: History is not accessible"):
                list_jobs_fn(history_id="h1")

    def test_an_unknown_history_is_an_empty_page_not_an_error(self, mock_galaxy_instance):
        # Galaxy filters the index by history_id without looking the history up, so a
        # well-formed id nothing answers to comes back 200 [] -- the same bytes as a history
        # with no jobs. The tool passes that through as success and says so in its
        # description, which is where a reconcile loop learns to confirm the history first.
        with self._connected(mock_galaxy_instance):
            mock_galaxy_instance.jobs._get.return_value = []
            result = list_jobs_fn(history_id="h404")
        assert result.success is True
        assert result.data == []
        assert result.count == 0
        assert result.message == "Retrieved 0 jobs"
        assert "get_history_details" in (list_jobs_fn.__doc__ or "")

    def test_a_history_the_user_cannot_read_carries_the_permission_hint(self, mock_galaxy_instance):
        # Galaxy's answer for a history the key cannot see: ItemAccessibilityException, 403.
        with self._connected(mock_galaxy_instance):
            mock_galaxy_instance.jobs._get.side_effect = bioblend.ConnectionError(
                "403 Client Error", body="Cannot access the request job objects.", status_code=403
            )
            with pytest.raises(ValueError) as excinfo:
                list_jobs_fn(history_id="h403")
        text = str(excinfo.value)
        assert text.startswith("List jobs failed: 403 Client Error")
        assert "(Permission denied - check your account permissions)" in text
        assert text.endswith("Context: history_id=h403")

    def test_a_malformed_history_id_says_what_the_400_means(self, mock_galaxy_instance):
        # A history_id Galaxy cannot decode is a 400 MalformedId. The shared hint table has
        # no 400 row, so the tool adds the one thing a caller needs: this is a filter Galaxy
        # could not read, not the missing-history answer, which is an empty page.
        with self._connected(mock_galaxy_instance):
            mock_galaxy_instance.jobs._get.side_effect = bioblend.ConnectionError(
                "400 Client Error", body="Wrong  id ( not-an-id ) specified", status_code=400
            )
            with pytest.raises(ValueError) as excinfo:
                list_jobs_fn(history_id="not-an-id")
        text = str(excinfo.value)
        assert text.startswith("List jobs failed: 400 Client Error")
        assert "Context: history_id=not-an-id. Galaxy could not read one of the filters" in text
        assert text.endswith("it answers with an empty page")

    def test_other_statuses_are_wrapped_without_the_filter_hint(self, mock_galaxy_instance):
        with self._connected(mock_galaxy_instance):
            mock_galaxy_instance.jobs._get.side_effect = bioblend.ConnectionError(
                "500 Server Error", body="boom", status_code=500
            )
            with pytest.raises(ValueError) as excinfo:
                list_jobs_fn(history_id="h1")
        text = str(excinfo.value)
        assert "(Server error - try again later or contact admin)" in text
        assert text.endswith("Context: history_id=h1")
