"""Tests for job operations"""

import re
from unittest.mock import Mock, patch

import bioblend
import pytest
import responses

from galaxy_mcp.server import _log_ends, galaxy_state
from tests.test_helpers import get_job_details_fn, get_job_logs_fn, list_jobs_fn


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

    def test_describes_only_what_the_plain_read_carries(self):
        # view_show_job adds job_metrics inside `if full:` and only for an admin, so the
        # plain read never has them; the description must not send a caller looking for a
        # field no call to this tool can return. The logs are named as get_job_logs' question
        # without naming its parameter, because the TypeScript surfaces respell a parameter
        # token in their own spelling and a command line has no --job-id on get_job_logs.
        doc = get_job_details_fn.__doc__ or ""
        assert "metrics" not in doc
        assert "one get_job_logs call away" in doc
        assert "get_job_logs(" not in doc

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


# The clamp's vectors, as literal expected strings. The TypeScript log-ends.test.ts holds the
# same table, and the multibyte golden case pins the two against each other byte for byte.
EURO_LINES = "\n".join("€" * 7 for _ in range(12))
LOG_ENDS_VECTORS = [
    ("short, untouched", "a\nb\n", 4096, "a\nb\n"),
    ("empty at any budget", "", 5, ""),
    ("budget 0 is uncut", "x" * 100, 0, "x" * 100),
    (
        "ASCII lines at the default budget",
        "\n".join(f"line {i:03d} of a long log" for i in range(300)),
        4096,
        "\n".join(f"line {i:03d} of a long log" for i in range(89))
        + "\n[... 2807 of 6899 bytes omitted ...]\n"
        + "\n".join(f"line {i:03d} of a long log" for i in range(211, 300)),
    ),
    (
        "3-byte text with no newline: both cuts land mid-character",
        "€" * 40,
        64,
        "€" * 10 + "\n[... 60 of 120 bytes omitted ...]\n" + "€" * 10,
    ),
    (
        "newline in each half, through multi-byte text",
        EURO_LINES,
        64,
        "€" * 7 + "\n[... 221 of 263 bytes omitted ...]\n" + "€" * 7,
    ),
    (
        "4-byte text with no newline",
        "🎉" * 50,
        64,
        "🎉" * 8 + "\n[... 136 of 200 bytes omitted ...]\n" + "🎉" * 8,
    ),
    (
        "mixed widths, and a tail that is empty because the newline is back's last byte",
        "a€b🎉c\n" * 30,
        11,
        "a€b\n[... 325 of 330 bytes omitted ...]\n",
    ),
    (
        "a BOM at the start of the tail is kept",
        "x" * 60 + "\ufeff" + "y" * 40,
        86,
        "x" * 43 + "\n[... 17 of 103 bytes omitted ...]\n" + "\ufeff" + "y" * 40,
    ),
    (
        "CRLF: the cut is on the newline, so the head ends in a bare CR",
        "\r\n".join(f"l{i}" for i in range(40)),
        32,
        "l0\r\nl1\r\nl2\r\nl3\r" + "\n[... 160 of 188 bytes omitted ...]\n" + "l37\r\nl38\r\nl39",
    ),
    ("budget 1: halves of 0, both ends empty", "abc\n", 1, "\n[... 4 of 4 bytes omitted ...]\n"),
    (
        "odd budget: one byte unused on each side",
        "a" * 100,
        33,
        "a" * 16 + "\n[... 68 of 100 bytes omitted ...]\n" + "a" * 16,
    ),
    (
        "a newline only at byte 0: the head is empty",
        "\n" + "b" * 100,
        20,
        "\n[... 91 of 101 bytes omitted ...]\n" + "b" * 10,
    ),
]


class TestLogEnds:
    @pytest.mark.parametrize(
        ("text", "budget", "expected"),
        [(text, budget, expected) for _, text, budget, expected in LOG_ENDS_VECTORS],
        ids=[name for name, *_ in LOG_ENDS_VECTORS],
    )
    def test_vectors(self, text, budget, expected):
        got = _log_ends(text, budget)

        assert got == expected
        assert "\ufffd" not in got

    def test_an_untouched_log_is_the_same_object(self):
        text = "short"
        assert _log_ends(text, 0) is text
        assert _log_ends(text, 4096) is text

    def test_a_lone_surrogate_counts_as_the_three_bytes_textencoder_writes(self):
        # json.loads turns a "\ud800" escape into a lone surrogate that str.encode refuses;
        # the other surface's TextEncoder writes EF BF BD for it, so the count has to agree.
        text = "\ud800" * 30
        assert (
            _log_ends(text, 12)
            == "\ufffd" * 2 + "\n[... 78 of 90 bytes omitted ...]\n" + "\ufffd" * 2
        )


class TestGetJobLogs:
    BASE = "http://localhost:8080"
    JOB_ID = "0123456789abcdef"

    def setup_method(self):
        galaxy_state["connected"] = True
        galaxy_state["gi"] = type("MockGI", (), {})()
        galaxy_state["url"] = "http://localhost:8080/"
        galaxy_state["api_key"] = "test_key"

    def teardown_method(self):
        galaxy_state["connected"] = False
        galaxy_state["gi"] = None

    @responses.activate
    def test_reads_the_full_job_and_answers_only_the_log_fields(self):
        responses.add(
            responses.GET,
            f"{self.BASE}/api/jobs/{self.JOB_ID}",
            json={
                "id": self.JOB_ID,
                "state": "error",
                "exit_code": 1,
                "tool_stdout": "a\nb\n",
                "tool_stderr": "",
                "job_stdout": "ran",
                "job_stderr": None,
                "job_metrics": [],
            },
            match=[responses.matchers.query_param_matcher({"full": "true"})],
        )

        result = get_job_logs_fn(self.JOB_ID)

        assert result.success is True
        assert result.data == {"tool_stdout": "a\nb\n", "tool_stderr": "", "job_stdout": "ran"}
        assert list(result.data) == ["tool_stdout", "tool_stderr", "job_stdout"]
        assert result.message == f"Retrieved job logs for job '{self.JOB_ID}'"
        assert result.count is None

    @responses.activate
    def test_a_job_that_has_not_run_answers_the_empty_streams_galaxy_always_sends(self):
        # view_show_job (lib/galaxy/managers/jobs.py) copies the four tool_*/job_* columns as
        # they are, null before the job has run, while Job.stdout and Job.stderr are
        # properties that always answer a string. So the only empty answer a Galaxy gives is
        # this one -- the two legacy fields as "" -- and never a record with no logs at all.
        responses.add(
            responses.GET,
            f"{self.BASE}/api/jobs/{self.JOB_ID}",
            json={
                "id": self.JOB_ID,
                "state": "new",
                "tool_stdout": None,
                "tool_stderr": None,
                "stdout": "",
                "stderr": "",
            },
        )

        result = get_job_logs_fn(self.JOB_ID)

        assert result.data == {"stdout": "", "stderr": ""}
        assert result.message == f"Retrieved job logs for job '{self.JOB_ID}'"
        doc = get_job_logs_fn.__doc__ or ""
        assert "always answers" in doc
        assert "has not written yet" not in doc

    @responses.activate
    def test_a_long_log_is_cut_to_the_budget_and_zero_leaves_it_whole(self):
        log = "\n".join(f"line {i}" for i in range(500))
        responses.add(
            responses.GET,
            f"{self.BASE}/api/jobs/{self.JOB_ID}",
            json={"id": self.JOB_ID, "stderr": log},
        )

        cut = get_job_logs_fn(self.JOB_ID, log_bytes=100).data["stderr"]
        whole = get_job_logs_fn(self.JOB_ID, log_bytes=0).data["stderr"]

        assert cut == _log_ends(log, 100)
        assert "bytes omitted ...]" in cut
        assert whole == log

    @responses.activate
    def test_a_negative_budget_is_refused_before_anything_is_sent(self):
        with pytest.raises(ValueError, match=re.escape("log_bytes must be 0 or greater (got -1)")):
            get_job_logs_fn(self.JOB_ID, log_bytes=-1)

        assert len(responses.calls) == 0

    @pytest.mark.parametrize("value", [".", "../histories", "not-an-id"])
    @responses.activate
    def test_a_value_that_is_not_an_id_is_refused_before_anything_is_sent(self, value):
        with pytest.raises(ValueError, match="not a Galaxy id"):
            get_job_logs_fn(value)

        assert len(responses.calls) == 0

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
            get_job_logs_fn(self.JOB_ID)

    @responses.activate
    def test_any_other_status_names_this_tool(self):
        responses.add(responses.GET, f"{self.BASE}/api/jobs/{self.JOB_ID}", status=500)

        with pytest.raises(ValueError, match=f"Get job logs.*job_id={self.JOB_ID}"):
            get_job_logs_fn(self.JOB_ID)

    @responses.activate
    def test_a_200_that_is_not_this_jobs_record_is_refused(self):
        responses.add(
            responses.GET,
            f"{self.BASE}/api/jobs/{self.JOB_ID}",
            json={"id": "fedcba9876543210", "tool_stderr": "x" * 10000},
        )

        with pytest.raises(ValueError, match="not that job's record"):
            get_job_logs_fn(self.JOB_ID)

    def test_not_connected(self):
        galaxy_state["connected"] = False

        with pytest.raises(ValueError, match="Not connected"):
            get_job_logs_fn(self.JOB_ID)


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
