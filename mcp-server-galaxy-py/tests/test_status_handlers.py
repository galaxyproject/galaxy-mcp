"""The five tools that decide "not found" now decide it from the status, not the text.

Each of these used to ask `"404" in str(e)`. requests quotes the URL it could not reach,
so a history or dataset id with 404 in it made every connection failure look like a
missing resource -- the caller was told to check an id that was fine, about a server that
never answered. `_http_status` reads the status bioblend and requests actually carry, and
`_is_not_found` is the only question the handlers ask.

These drive a real bioblend client and real requests against `responses` rather than
mocking the exception, because the shape of the failure is the thing under test: a 404
carries an integer, and bioblend's GET path substitutes an empty Response for a connection
error, which leaves the status None and the text requests' own. A mocked exception would
prove only that the test author agreed with themselves.

Every id below has 404 in it on purpose.
"""

import pytest
import requests
import responses
from bioblend.galaxy import GalaxyInstance

from galaxy_mcp.server import galaxy_state
from tests.test_helpers import (
    get_collection_details_fn,
    get_dataset_details_fn,
    get_history_contents_fn,
    get_history_details_fn,
    get_job_details_fn,
)

BASE = "http://status-test.invalid"

# The id is the trap: requests puts the URL in the text of a failure it never completed,
# and the tool's own wrapper sentences quote it too.
ID_404 = "404aaaaaaaaaaaaa"
ID_401 = "401aaaaaaaaaaaaa"
HISTORY_404 = "404bbbbbbbbbbbbb"

UNREACHED = (
    f"HTTPConnectionPool(host='status-test.invalid', port=80): Max retries exceeded "
    f"with url: /api/datasets/{ID_404}"
)

NOT_FOUND = "not found"
SERVER_ERROR_HINT = "Server error"
AUTH_HINT = "Authentication failed"

# Every hint format_error can append, for the tests that assert none of them was.
EVERY_HINT = (AUTH_HINT, "Permission denied", "Resource not found", SERVER_ERROR_HINT)


@pytest.fixture(autouse=True)
def connected():
    """Point the server at a Galaxy that only exists inside `responses`."""
    gi = GalaxyInstance(url=BASE, key="notakey")
    galaxy_state.update({"connected": True, "gi": gi, "url": f"{BASE}/", "api_key": "notakey"})
    yield gi
    galaxy_state.update({"connected": False, "gi": None, "url": None, "api_key": None})


def answer(url: str, status: int) -> None:
    responses.add(responses.GET, url, json={"err_msg": f"error {status}"}, status=status)


def never_answers(url: str, text: str = UNREACHED) -> None:
    """A request that does not complete, whose text quotes a URL with 404 in it."""
    responses.add(responses.GET, url, body=requests.ConnectionError(text))


def failure_from(call) -> str:
    with pytest.raises(ValueError) as raised:
        call()
    return str(raised.value)


class TestGetHistoryDetails:
    URL = f"{BASE}/api/histories/{HISTORY_404}"

    @responses.activate
    def test_a_real_404_is_a_missing_history(self):
        answer(self.URL, 404)

        message = failure_from(lambda: get_history_details_fn(HISTORY_404))

        assert f"History ID '{HISTORY_404}' not found" in message

    @responses.activate
    def test_a_request_that_never_completed_claims_nothing_about_the_id(self):
        never_answers(self.URL)

        message = failure_from(lambda: get_history_details_fn(HISTORY_404))

        assert NOT_FOUND not in message.lower()
        assert "Get history details failed" in message
        assert "Max retries exceeded" in message

    @responses.activate
    def test_a_500_is_reported_as_the_server_s_problem(self):
        answer(self.URL, 500)

        message = failure_from(lambda: get_history_details_fn(HISTORY_404))

        assert SERVER_ERROR_HINT in message
        assert NOT_FOUND not in message.lower()


class TestGetHistoryContents:
    URL = f"{BASE}/api/histories/{HISTORY_404}/contents"

    @responses.activate
    def test_a_real_404_is_a_missing_history(self):
        answer(self.URL, 404)

        message = failure_from(lambda: get_history_contents_fn(HISTORY_404))

        assert f"History ID '{HISTORY_404}' not found" in message

    @responses.activate
    def test_a_request_that_never_completed_claims_nothing_about_the_id(self):
        never_answers(self.URL)

        message = failure_from(lambda: get_history_contents_fn(HISTORY_404))

        assert NOT_FOUND not in message.lower()
        assert "Get history contents failed" in message

    @responses.activate
    def test_a_500_is_reported_as_the_server_s_problem(self):
        answer(self.URL, 500)

        message = failure_from(lambda: get_history_contents_fn(HISTORY_404))

        assert SERVER_ERROR_HINT in message
        assert NOT_FOUND not in message.lower()


class TestGetDatasetDetails:
    """The dataset read has a second question in its handler: is this id a collection?

    Both calls are answered here, because a collection lookup that succeeded would take a
    different branch, and one that is not registered at all fails in a way `responses`
    chose rather than the test.
    """

    URL = f"{BASE}/api/datasets/{ID_404}"
    COLLECTION_URL = f"{BASE}/api/dataset_collections/{ID_404}"

    @responses.activate
    def test_a_real_404_is_a_missing_dataset(self):
        answer(self.URL, 404)
        answer(self.COLLECTION_URL, 404)

        message = failure_from(lambda: get_dataset_details_fn(ID_404))

        assert f"Dataset ID '{ID_404}' not found" in message

    @responses.activate
    def test_a_request_that_never_completed_claims_nothing_about_the_id(self):
        never_answers(self.URL)
        never_answers(self.COLLECTION_URL)

        message = failure_from(lambda: get_dataset_details_fn(ID_404))

        assert NOT_FOUND not in message.lower()
        assert "Get dataset details failed" in message

    @responses.activate
    def test_a_500_is_reported_as_the_server_s_problem(self):
        answer(self.URL, 500)
        answer(self.COLLECTION_URL, 500)

        message = failure_from(lambda: get_dataset_details_fn(ID_404))

        assert SERVER_ERROR_HINT in message
        assert NOT_FOUND not in message.lower()


class TestGetCollectionDetails:
    URL = f"{BASE}/api/dataset_collections/{ID_404}"

    @responses.activate
    def test_a_real_404_is_a_missing_collection(self):
        answer(self.URL, 404)

        message = failure_from(lambda: get_collection_details_fn(ID_404))

        assert f"Collection ID '{ID_404}' not found" in message

    @responses.activate
    def test_a_request_that_never_completed_claims_nothing_about_the_id(self):
        never_answers(self.URL)

        message = failure_from(lambda: get_collection_details_fn(ID_404))

        assert NOT_FOUND not in message.lower()
        assert "Get collection details failed" in message

    @responses.activate
    def test_a_500_is_reported_as_the_server_s_problem(self):
        answer(self.URL, 500)

        message = failure_from(lambda: get_collection_details_fn(ID_404))

        assert SERVER_ERROR_HINT in message
        assert NOT_FOUND not in message.lower()


class TestGetJobDetails:
    """The one that does not go through bioblend for the call that fails.

    Provenance comes from bioblend and answers; the job read is a plain `requests.get`
    followed by `raise_for_status`, so the failure carries its status on `response` rather
    than on a field of its own. Same question, other spelling.
    """

    JOB_ID = "404ccccccccccccc"
    PROVENANCE_URL = f"{BASE}/api/histories/{HISTORY_404}/contents/{ID_404}/provenance"
    JOB_URL = f"{BASE}/api/jobs/{JOB_ID}"

    def _provenance(self):
        responses.add(responses.GET, self.PROVENANCE_URL, json={"job_id": self.JOB_ID}, status=200)

    @responses.activate
    def test_a_real_404_is_a_missing_or_unreadable_job(self):
        self._provenance()
        answer(self.JOB_URL, 404)

        message = failure_from(lambda: get_job_details_fn(ID_404, history_id=HISTORY_404))

        assert f"Dataset ID '{ID_404}' not found or job not accessible" in message

    @responses.activate
    def test_a_request_that_never_completed_claims_nothing_about_the_id(self):
        self._provenance()
        never_answers(self.JOB_URL)

        message = failure_from(lambda: get_job_details_fn(ID_404, history_id=HISTORY_404))

        assert NOT_FOUND not in message.lower()
        assert "not accessible" not in message
        assert "Get job details failed" in message

    @responses.activate
    def test_a_500_is_reported_as_the_server_s_problem(self):
        self._provenance()
        answer(self.JOB_URL, 500)

        message = failure_from(lambda: get_job_details_fn(ID_404, history_id=HISTORY_404))

        assert SERVER_ERROR_HINT in message
        assert NOT_FOUND not in message.lower()


class TestGetJobDetailsFindingTheJob:
    """The two lookups before the job read, whose failures used to be folded into a sentence
    of the tool's own before anything asked what had gone wrong.

    That wrapper carried no status, so the fallback searched its text -- and its text quotes
    the dataset id. `401aaaaaaaaaaaaa` was answered with an API key to check, and so was a
    dataset lookup that really did return 500. A real 404 wrapped the same way lost the
    sentence about permission that is the reason this tool writes its own message at all.
    """

    DATASET_URL = f"{BASE}/api/datasets/{ID_401}"

    @responses.activate
    def test_a_dataset_no_job_made_is_not_a_rejected_api_key(self):
        """Nothing failed over HTTP, so there is no status to have an opinion about."""
        responses.add(responses.GET, self.DATASET_URL, json={"id": ID_401}, status=200)

        message = failure_from(lambda: get_job_details_fn(ID_401))

        assert "No job information found" in message
        assert not [hint for hint in EVERY_HINT if hint in message]

    @responses.activate
    def test_a_500_from_the_dataset_lookup_is_the_server_s_problem(self):
        answer(self.DATASET_URL, 500)

        message = failure_from(lambda: get_job_details_fn(ID_401))

        assert SERVER_ERROR_HINT in message
        assert AUTH_HINT not in message

    @responses.activate
    def test_a_404_from_the_dataset_lookup_still_says_it_may_be_permission(self):
        answer(self.DATASET_URL, 404)

        message = failure_from(lambda: get_job_details_fn(ID_401))

        assert f"Dataset ID '{ID_401}' not found or job not accessible" in message
        assert "permission to view it" in message
        assert AUTH_HINT not in message


class TestGetJobDetailsWhenProvenanceFailed:
    """Provenance failed, the dataset record answered, and it knows of no job.

    The one path where two lookups disagree about what happened. "The dataset was not
    created by a job" is what the fallback found; it is not an answer to why provenance
    failed, and saying it instead throws away the only exception here that knows. The
    caller asked about provenance -- a 500, a permission refusal or a request that never
    completed has to survive a fallback that came back empty.

    The dataset id carries 401 so that anything reading a status out of the text would say
    the API key was rejected, and every one of these failures is something else.
    """

    PROVENANCE_URL = f"{BASE}/api/histories/{HISTORY_404}/contents/{ID_401}/provenance"
    DATASET_URL = f"{BASE}/api/datasets/{ID_401}"
    JOBLESS = "may not have been created by a job"
    UNREACHED_PROVENANCE = (
        f"HTTPConnectionPool(host='status-test.invalid', port=80): Max retries exceeded "
        f"with url: /api/histories/{HISTORY_404}/contents/{ID_401}/provenance"
    )

    def _dataset_without_a_job(self):
        responses.add(responses.GET, self.DATASET_URL, json={"id": ID_401}, status=200)

    @responses.activate
    def test_a_500_from_provenance_survives_a_fallback_that_found_no_job(self):
        answer(self.PROVENANCE_URL, 500)
        self._dataset_without_a_job()

        message = failure_from(lambda: get_job_details_fn(ID_401, history_id=HISTORY_404))

        assert SERVER_ERROR_HINT in message
        assert "error 500" in message
        assert self.JOBLESS not in message
        assert AUTH_HINT not in message

    @responses.activate
    def test_a_404_from_provenance_still_says_it_may_be_permission(self):
        answer(self.PROVENANCE_URL, 404)
        self._dataset_without_a_job()

        message = failure_from(lambda: get_job_details_fn(ID_401, history_id=HISTORY_404))

        assert f"Dataset ID '{ID_401}' not found or job not accessible" in message
        assert "permission to view it" in message
        assert self.JOBLESS not in message
        assert AUTH_HINT not in message

    @responses.activate
    def test_provenance_that_never_completed_is_reported_without_a_diagnosis(self):
        never_answers(self.PROVENANCE_URL, self.UNREACHED_PROVENANCE)
        self._dataset_without_a_job()

        message = failure_from(lambda: get_job_details_fn(ID_401, history_id=HISTORY_404))

        assert "Get job details failed" in message
        assert "Max retries exceeded" in message
        assert self.JOBLESS not in message
        assert NOT_FOUND not in message.lower()
        assert not [hint for hint in EVERY_HINT if hint in message]
