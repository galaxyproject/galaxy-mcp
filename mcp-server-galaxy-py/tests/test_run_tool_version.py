"""run_tool can ask for a specific tool version, which bioblend's run_tool cannot.

bioblend's ToolClient.run_tool takes history_id, tool_id, tool_inputs, input_format,
data_manager_mode and credentials_context -- no tool_version. Galaxy does accept one:
services/tools.py _create reads payload.get("tool_version") into a ToolRunReference. So
when a version is asked for, the payload bioblend would have built is built here and
posted the same way, and these tests hold the two payloads against each other so the
hand-built one cannot drift.

Asking is not getting, and that runs through the rest of this file. Galaxy's toolbox
returns the newest installed version when the requested one is not there
(v26.1.1 tool_util/toolbox/base.py get_tool), so the version in the request is not
evidence of the version that ran, and the schema for one version is not a description
of another.
"""

import json
from urllib.parse import parse_qs, urlparse

import pytest
import responses
from bioblend.galaxy import GalaxyInstance

from galaxy_mcp.server import galaxy_state
from tests.test_helpers import run_tool_fn

GALAXY_BASE_URL = "http://run-tool-test.invalid"
HISTORY_ID = "1cd8e2f6b131e5aa"
TOOL_ID = "cat1"
INPUTS = {"input1": {"src": "hda", "id": "f2db41e1fa331b3e"}}
COLLECTION_INPUTS = {"input1": {"src": "hdca", "id": "c0ffeec0ffeec0ff"}}

ONE_DATASET = {"name": "input1", "type": "data", "multiple": False}
ONE_COLLECTION = {"name": "input1", "type": "data_collection"}


def _schema(version, param):
    """What GET /api/tools/{id}?io_details=True answers with."""
    return {"id": TOOL_ID, "version": version, "inputs": [param]}


# What the toolbox has installed, and what an unversioned lookup therefore resolves to.
INSTALLED = _schema("2.0.0", ONE_DATASET)
# The older version, which took a collection where the installed one takes one dataset.
V1 = _schema("1.0.0", ONE_COLLECTION)


def _jobs(*versions):
    """A reply with one job per argument; None is a job that names no version."""
    jobs = []
    for index, version in enumerate(versions):
        job = {"id": f"job{index}", "state": "new"}
        if version is not None:
            job["tool_version"] = version
        jobs.append(job)
    return {"jobs": jobs, "outputs": []}


JOB = _jobs("1.0.2")


@pytest.fixture
def connected(monkeypatch):
    """A real GalaxyInstance against a URL that only exists inside `responses`."""
    gi = GalaxyInstance(url=GALAXY_BASE_URL, key="notakey")
    monkeypatch.setattr("galaxy_mcp.server._get_tool_credentials_context", lambda *a, **k: None)
    galaxy_state.update(
        {"connected": True, "gi": gi, "url": f"{GALAXY_BASE_URL}/", "api_key": "notakey"}
    )
    return gi


def _serve_schemas(default, by_version=None):
    """Answer the preflight's schema lookup, picking by the tool_version asked for.

    Returns the list of versions asked for, in order, so a test can prove both that
    the version travelled and that the cache did not answer for a different one.
    ``by_version`` missing an entry is Galaxy falling back to what it has installed.
    """
    asked_for: list[str | None] = []

    def handler(request):
        asked = parse_qs(urlparse(request.url).query).get("tool_version", [None])[0]
        asked_for.append(asked)
        body = (by_version or {}).get(asked, default)
        return (200, {"Content-Type": "application/json"}, json.dumps(body))

    responses.add_callback(
        responses.GET,
        f"{GALAXY_BASE_URL}/api/tools/{TOOL_ID}",
        callback=handler,
        content_type="application/json",
    )
    return asked_for


def _posted_body():
    posts = [c for c in responses.calls if c.request.method == "POST"]
    assert len(posts) == 1, [c.request.url for c in responses.calls]
    return json.loads(posts[0].request.body)


class TestTheVersionReachesGalaxy:
    @responses.activate
    def test_a_version_is_posted_in_the_payload(self, connected):
        _serve_schemas(INSTALLED, {"1.0.2": _schema("1.0.2", ONE_DATASET)})
        responses.add(responses.POST, f"{GALAXY_BASE_URL}/api/tools", json=JOB)

        result = run_tool_fn(HISTORY_ID, TOOL_ID, INPUTS, tool_version="1.0.2")

        body = _posted_body()
        assert body["tool_version"] == "1.0.2"
        assert body["tool_id"] == TOOL_ID
        assert body["history_id"] == HISTORY_ID
        assert body["inputs"] == INPUTS
        assert body["input_format"] == "legacy"
        assert result.success is True
        assert result.data == JOB
        assert "not pre-checked" not in result.message

    @responses.activate
    def test_no_version_means_no_key_and_galaxy_chooses(self, connected):
        """Omitting it has to leave the payload exactly as it was, not send a null."""
        _serve_schemas(INSTALLED)
        responses.add(responses.POST, f"{GALAXY_BASE_URL}/api/tools", json=JOB)

        result = run_tool_fn(HISTORY_ID, TOOL_ID, INPUTS)

        assert "tool_version" not in _posted_body()
        assert result.success is True
        assert "version" not in result.message

    @responses.activate
    def test_the_payload_is_bioblends_plus_the_version(self, connected):
        """Captured from bioblend itself, so the hand-built one cannot drift from it.

        The version path does not go through bioblend, so nothing else would notice if
        bioblend started sending another key -- credentials_context arrived that way.
        """
        _serve_schemas(INSTALLED, {"1.0.2": _schema("1.0.2", ONE_DATASET)})
        responses.add(responses.POST, f"{GALAXY_BASE_URL}/api/tools", json=JOB)
        connected.tools.run_tool(HISTORY_ID, TOOL_ID, INPUTS)
        from_bioblend = _posted_body()

        responses.calls.reset()
        run_tool_fn(HISTORY_ID, TOOL_ID, INPUTS, tool_version="1.0.2")
        ours = _posted_body()

        assert ours == {**from_bioblend, "tool_version": "1.0.2"}


class TestTheMessageReportsTheVersionThatRan:
    """The request is not the answer, so the message may not repeat it as one.

    POST /api/tools serialises each job through Job.to_dict(view="collection"), whose
    visible keys include tool_version, so the job record says which version ran. When
    it does not say, neither do we -- and one job out of several not saying is the
    same thing, because half a provenance is not one.
    """

    @responses.activate
    def test_the_job_version_is_what_the_message_says(self, connected):
        _serve_schemas(INSTALLED, {"1.0.2": _schema("1.0.2", ONE_DATASET)})
        responses.add(responses.POST, f"{GALAXY_BASE_URL}/api/tools", json=_jobs("1.0.2"))

        result = run_tool_fn(HISTORY_ID, TOOL_ID, INPUTS, tool_version="1.0.2")

        assert "at version 1.0.2" in result.message
        assert "requested" not in result.message

    @responses.activate
    def test_a_job_that_ran_another_version_is_reported_as_that(self, connected):
        """The toolbox fell back, so claiming 1.0.2 ran would be a fabricated record."""
        _serve_schemas(INSTALLED, {"1.0.2": _schema("1.0.2", ONE_DATASET)})
        responses.add(responses.POST, f"{GALAXY_BASE_URL}/api/tools", json=_jobs("2.0.0"))

        result = run_tool_fn(HISTORY_ID, TOOL_ID, INPUTS, tool_version="1.0.2")

        assert "at version 2.0.0 (not the 1.0.2 requested)" in result.message
        assert result.data["jobs"][0]["tool_version"] == "2.0.0"

    @responses.activate
    def test_a_reply_that_names_no_version_claims_none(self, connected):
        _serve_schemas(INSTALLED, {"1.0.2": _schema("1.0.2", ONE_DATASET)})
        responses.add(responses.POST, f"{GALAXY_BASE_URL}/api/tools", json=_jobs(None))

        result = run_tool_fn(HISTORY_ID, TOOL_ID, INPUTS, tool_version="1.0.2")

        assert "at an unreported version (1.0.2 requested)" in result.message
        assert "at version 1.0.2" not in result.message

    @responses.activate
    def test_one_job_of_several_without_a_version_reports_none(self, connected):
        """A version for some of the jobs is not a version for the submission.

        A batch run makes several jobs from one call. Picking the version out of the
        ones that named it would report a run at 1.0.2 whose other job might have
        been anything.
        """
        _serve_schemas(INSTALLED, {"1.0.2": _schema("1.0.2", ONE_DATASET)})
        responses.add(
            responses.POST,
            f"{GALAXY_BASE_URL}/api/tools",
            json={
                "jobs": [
                    {"id": "job0", "state": "new", "tool_version": "1.0.2"},
                    {"id": "job1", "state": "new", "tool_version": None},
                ],
                "outputs": [],
            },
        )

        result = run_tool_fn(HISTORY_ID, TOOL_ID, INPUTS, tool_version="1.0.2")

        assert "at an unreported version (1.0.2 requested)" in result.message
        assert "at version 1.0.2" not in result.message

    @responses.activate
    def test_jobs_that_disagree_report_none(self, connected):
        _serve_schemas(INSTALLED, {"1.0.2": _schema("1.0.2", ONE_DATASET)})
        responses.add(responses.POST, f"{GALAXY_BASE_URL}/api/tools", json=_jobs("1.0.2", "2.0.0"))

        result = run_tool_fn(HISTORY_ID, TOOL_ID, INPUTS, tool_version="1.0.2")

        assert "at an unreported version (1.0.2 requested)" in result.message
        assert "2.0.0" not in result.message

    @responses.activate
    def test_a_reply_with_no_jobs_reports_none(self, connected):
        """Nothing ran that said anything, so there is nothing to say."""
        _serve_schemas(INSTALLED, {"1.0.2": _schema("1.0.2", ONE_DATASET)})
        responses.add(responses.POST, f"{GALAXY_BASE_URL}/api/tools", json=_jobs())

        result = run_tool_fn(HISTORY_ID, TOOL_ID, INPUTS, tool_version="1.0.2")

        assert "at an unreported version (1.0.2 requested)" in result.message


class TestThePreflightChecksTheVersionBeingRun:
    """A run pinned to v1 has to be checked against v1's parameters, not the default's.

    The input check fetches the tool's io_details schema, and an id on its own
    resolves to whatever is installed. Here v1 takes a collection and the installed
    2.0.0 takes one dataset, so checking a v1 run against the installed schema would
    refuse inputs that are exactly right for the version being run.
    """

    @responses.activate
    def test_v1_inputs_pass_when_v1_is_the_version_asked_for(self, connected):
        asked_for = _serve_schemas(INSTALLED, {"1.0.0": V1})
        responses.add(responses.POST, f"{GALAXY_BASE_URL}/api/tools", json=_jobs("1.0.0"))

        result = run_tool_fn(HISTORY_ID, TOOL_ID, COLLECTION_INPUTS, tool_version="1.0.0")

        assert result.success is True
        assert "not pre-checked" not in result.message
        assert asked_for == ["1.0.0"]
        assert _posted_body()["inputs"] == COLLECTION_INPUTS

    @responses.activate
    def test_the_same_inputs_are_still_refused_for_the_installed_version(self, connected):
        """The check is real: without the version it is the installed schema that applies."""
        _serve_schemas(INSTALLED, {"1.0.0": V1})
        responses.add(responses.POST, f"{GALAXY_BASE_URL}/api/tools", json=_jobs("2.0.0"))

        with pytest.raises(ValueError, match="not submitting"):
            run_tool_fn(HISTORY_ID, TOOL_ID, COLLECTION_INPUTS)

        assert not [c for c in responses.calls if c.request.method == "POST"]

    @responses.activate
    def test_the_cache_does_not_cross_versions(self, connected):
        asked_for = _serve_schemas(INSTALLED, {"1.0.0": V1})
        responses.add(responses.POST, f"{GALAXY_BASE_URL}/api/tools", json=_jobs("1.0.0"))

        run_tool_fn(HISTORY_ID, TOOL_ID, COLLECTION_INPUTS, tool_version="1.0.0")
        run_tool_fn(HISTORY_ID, TOOL_ID, COLLECTION_INPUTS, tool_version="1.0.0")
        assert asked_for == ["1.0.0"], "the second v1 run should read v1's cache entry"

        with pytest.raises(ValueError, match="not submitting"):
            run_tool_fn(HISTORY_ID, TOOL_ID, COLLECTION_INPUTS)

        assert asked_for == ["1.0.0", None], "v1's entry is not an answer about the default"

    @responses.activate
    def test_a_version_galaxy_does_not_have_leaves_the_inputs_unchecked(self, connected):
        """The fallback comes back describing another version, which proves nothing."""
        _serve_schemas(INSTALLED)
        responses.add(responses.POST, f"{GALAXY_BASE_URL}/api/tools", json=_jobs("2.0.0"))

        result = run_tool_fn(HISTORY_ID, TOOL_ID, COLLECTION_INPUTS, tool_version="1.0.0")

        assert result.success is True
        assert "not pre-checked" in result.message
        assert "described version 2.0.0" in result.message
        assert "1.0.0 asked for" in result.message


class TestTheRestIsUnchanged:
    @responses.activate
    def test_a_failure_still_reports_through_the_standard_format(self, connected):
        _serve_schemas(INSTALLED, {"1.0.2": _schema("1.0.2", ONE_DATASET)})
        responses.add(responses.POST, f"{GALAXY_BASE_URL}/api/tools", status=400, json={})

        with pytest.raises(ValueError, match="Run tool failed"):
            run_tool_fn(HISTORY_ID, TOOL_ID, INPUTS, tool_version="1.0.2")
