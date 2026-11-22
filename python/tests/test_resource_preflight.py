from unittest.mock import patch

import pytest

from galaxy_mcp.resource_preflight import MIB, evaluate_resource_preflight

from .test_helpers import evaluate_user_tool_preflight_fn, galaxy_state

REPRESENTATION = {
    "requirements": [
        {"type": "resource", "cores_min": 2, "ram_min": 512, "ram_max": 4096, "timelimit": 300}
    ]
}
METRICS = [
    {"plugin": "cgroup", "name": "memory.peak", "raw_value": str(800 * MIB)},
    {"plugin": "core", "name": "runtime_seconds", "raw_value": "240"},
]


def test_recommends_resources_with_margin():
    result = evaluate_resource_preflight(REPRESENTATION, {"id": "job", "state": "ok"}, METRICS)
    assert result["recommendation"] == {"ram_min": 1200, "timelimit": 360}
    assert {warning["code"] for warning in result["warnings"]} == {
        "ram_min_underestimated",
        "timelimit_underestimated",
    }


def test_warns_when_metrics_are_unavailable():
    result = evaluate_resource_preflight({}, {"id": "job", "state": "error"}, [])
    assert {warning["code"] for warning in result["warnings"]} == {
        "peak_memory_unavailable",
        "runtime_unavailable",
        "test_run_failed",
        "resource_requirement_missing",
    }


def test_rejects_an_unfinished_job():
    with pytest.raises(ValueError, match="not finished"):
        evaluate_resource_preflight(REPRESENTATION, {"id": "job", "state": "running"}, [])


def test_mcp_reads_matching_tool_job_and_metrics(mock_galaxy_instance):
    gi = mock_galaxy_instance
    gi.url = "http://localhost:8080/api"
    job_id = "0123456789abcdef"
    responses = [
        {"tool_id": "resource_tool", "representation": REPRESENTATION},
        {"id": job_id, "tool_id": "resource_tool", "state": "ok"},
        METRICS,
    ]
    gi.make_get_request.side_effect = [
        type(
            "Response",
            (),
            {"raise_for_status": lambda self: None, "json": lambda self, value=value: value},
        )()
        for value in responses
    ]
    with patch.dict(galaxy_state, {"connected": True, "gi": gi}):
        result = evaluate_user_tool_preflight_fn("uuid", job_id)
    assert result.data["recommendation"]["ram_min"] == 1200
    assert [call.args[0] for call in gi.make_get_request.call_args_list] == [
        "http://localhost:8080/api/unprivileged_tools/uuid",
        f"http://localhost:8080/api/jobs/{job_id}",
        f"http://localhost:8080/api/jobs/{job_id}/metrics",
    ]


def test_mcp_rejects_a_job_from_another_tool(mock_galaxy_instance):
    gi = mock_galaxy_instance
    gi.url = "http://localhost:8080/api"
    job_id = "0123456789abcdef"
    responses = [
        {"tool_id": "resource_tool", "representation": REPRESENTATION},
        {"id": job_id, "tool_id": "other_tool", "state": "ok"},
    ]
    gi.make_get_request.side_effect = [
        type(
            "Response",
            (),
            {"raise_for_status": lambda self: None, "json": lambda self, value=value: value},
        )()
        for value in responses
    ]
    with patch.dict(galaxy_state, {"connected": True, "gi": gi}):
        with pytest.raises(ValueError, match="did not run"):
            evaluate_user_tool_preflight_fn("uuid", job_id)
