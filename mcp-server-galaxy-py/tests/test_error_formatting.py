"""format_error decides its status hint from the status, not from the error's text.

The hint this adds is a sentence a caller reads and acts on, so it has to be right or
absent. bioblend puts the HTTP status on its ConnectionError as a field, and when a GET
raises a requests ConnectionError it substitutes an empty Response, which leaves that
field None and the message requests' own -- and requests quotes the URL in it. (Other
requests errors, and any failure on a POST or PUT, propagate without the field at all.)
Reading a status out of that text told callers to check their ids for a failure that had
nothing to do with them.

A field of None says only that there is no status: it covers a connection never made and
a reply whose body failed partway through, so nothing can be concluded from it and
nothing is.
"""

import bioblend
import pytest
import requests

from galaxy_mcp.server import format_error

# An id with 404 in it is the whole point: requests quotes the URL it could not reach,
# so the text of the failure contains the caller's own identifier.
DATASET_ID = "404aaaaaaaaaaaaa"
UNREACHED = (
    "HTTPConnectionPool(host='galaxy.invalid', port=80): Max retries exceeded "
    f"with url: /api/datasets/{DATASET_ID}"
)


class TestWhenTheErrorCarriesAStatus:
    @pytest.mark.parametrize(
        ("status", "hint"),
        [
            (401, "Authentication failed"),
            (403, "Permission denied"),
            (404, "Resource not found"),
            (500, "Server error"),
        ],
    )
    def test_the_hint_comes_from_the_status(self, status, hint):
        error = bioblend.ConnectionError(
            f"Unexpected HTTP status code: {status}", status_code=status
        )

        assert hint in format_error("Do a thing", error)

    def test_a_status_with_no_hint_gets_none(self):
        error = bioblend.ConnectionError("Unexpected HTTP status code: 409", status_code=409)

        message = format_error("Do a thing", error)

        assert message == "Do a thing failed: Unexpected HTTP status code: 409: None"

    def test_the_text_cannot_override_the_status(self):
        """A 403 whose body happens to mention 401 is still a permission problem.

        401 is the status the text search tried first, so this is the case a fallback to
        the text would get wrong.
        """
        error = bioblend.ConnectionError(
            "Unexpected HTTP status code: 403", body="token 401 in the body", status_code=403
        )

        message = format_error("Do a thing", error)

        assert "Permission denied" in message
        assert "Authentication failed" not in message


class TestWhenTheFieldHoldsNoStatus:
    def test_nothing_is_added_and_no_reach_failure_is_claimed(self):
        """A None status is bioblend's empty stand-in Response, and it means only that.

        It is the same value for a connection never made and for a reply Galaxy did send
        whose body failed partway through -- requests' SSLError and its plain
        ConnectionError land on the same branch of bioblend's GET path. So there is
        nothing to say, and saying the server could not be reached would be the same
        class of wrong answer this helper exists to stop making.
        """
        error = bioblend.ConnectionError(UNREACHED, status_code=None)

        message = format_error("Get dataset details", error, {"dataset_id": DATASET_ID})

        assert "Resource not found" not in message
        assert "check IDs and URLs" not in message
        assert "could not be reached" not in message
        # Nothing appended between the text and the context, which is the real assertion.
        expected = f"Get dataset details failed: {error}. Context: dataset_id={DATASET_ID}"
        assert message == expected
        assert "Max retries exceeded" in message


class TestWhenRequestsKeptTheReply:
    """The other spelling of the same fact, for the calls that do not go through bioblend.

    `get_job_details` asks the jobs API with requests directly, so what reaches here is a
    requests exception: no `status_code` field at all, the reply on `response` instead, and
    `response` set to None when the request never completed. Both are the same two facts
    bioblend spells with one field, and both are read off the failure rather than its text.
    """

    @pytest.mark.parametrize(
        ("status", "hint"),
        [(401, "Authentication failed"), (404, "Resource not found"), (500, "Server error")],
    )
    def test_the_hint_comes_from_the_reply(self, status, hint):
        # The text says nothing about a status on purpose: requests usually repeats it there,
        # and a test whose text repeats it would pass just as well against the search this
        # replaced. The reply is the only place the status is written down here.
        reply = requests.Response()
        reply.status_code = status
        error = requests.HTTPError("Error for url: /api/jobs/x", response=reply)

        assert hint in format_error("Do a thing", error)

    def test_a_request_that_never_completed_claims_nothing(self):
        """`response` is None, which says there was no reply -- not that there was a 404."""
        error = requests.ConnectionError(UNREACHED)

        message = format_error("Get job details", error, {"dataset_id": DATASET_ID})

        assert "Resource not found" not in message
        expected = f"Get job details failed: {error}. Context: dataset_id={DATASET_ID}"
        assert message == expected


class TestWhenThereIsNoStatusToRead:
    def test_a_plain_exception_still_reads_its_text(self):
        """Nothing else to go on, so this half is unchanged."""
        assert "Resource not found" in format_error("Do a thing", Exception("boom 404"))
        assert "Authentication failed" in format_error("Do a thing", Exception("boom 401"))
        assert "Permission denied" in format_error("Do a thing", Exception("boom 403"))
        assert "Server error" in format_error("Do a thing", Exception("boom 500"))

    def test_the_first_status_in_the_text_still_wins(self):
        """The old chain tried 401, 403, 404, 500 in that order and stopped."""
        message = format_error("Do a thing", Exception("401 and 404"))

        assert "Authentication failed" in message
        assert "Resource not found" not in message

    def test_a_plain_exception_with_no_status_gets_no_hint(self):
        assert format_error("Do a thing", Exception("boom")) == "Do a thing failed: boom"

    def test_a_status_field_that_is_not_a_number_falls_back_to_the_text(self):
        """Not a status this understands, so it is treated as no status field at all."""
        error = bioblend.ConnectionError("Unexpected HTTP status code: 404")
        error.status_code = "404"  # type: ignore[assignment]

        assert "Resource not found" in format_error("Do a thing", error)
