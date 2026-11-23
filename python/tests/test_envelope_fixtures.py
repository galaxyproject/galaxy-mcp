"""The checked-in golden envelopes still describe what the tools return.

The files under ``contract/envelopes`` are the contract the TypeScript MCP server and
CLI are held to. They are only worth anything while they say what this server actually
emits, so this regenerates every case and compares. A tool whose envelope changes fails
here first -- before the other language's replay -- with the command to regenerate.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from .envelope_fixtures import (
    FIXTURE_ROOT,
    REGENERATE_COMMAND,
    Case,
    _json_body_matcher,
    _same_json,
    build,
    route,
    run_case,
)
from .test_helpers import invoke_workflow_fn


@pytest.fixture(scope="module")
def generated() -> dict[str, str]:
    return build()


def test_every_case_is_written_down(generated: dict[str, str]) -> None:
    missing = sorted(name for name in generated if not (FIXTURE_ROOT / name).exists())
    assert not missing, (
        f"no golden file for {', '.join(missing)}; run `{REGENERATE_COMMAND}` from python/"
    )


def test_nothing_stale_is_left_behind(generated: dict[str, str]) -> None:
    on_disk = {str(path.relative_to(FIXTURE_ROOT)) for path in FIXTURE_ROOT.rglob("*.json")}
    extra = sorted(on_disk - set(generated))
    assert not extra, (
        f"{', '.join(extra)} belongs to no case any more; run `{REGENERATE_COMMAND}` from python/"
    )


def test_the_envelopes_still_match(generated: dict[str, str]) -> None:
    stale = []
    for name, text in sorted(generated.items()):
        path = FIXTURE_ROOT / name
        if path.exists() and path.read_text() != text:
            stale.append(name)
    assert not stale, (
        f"{', '.join(stale)} no longer match what the server returns; run "
        f"`{REGENERATE_COMMAND}` from python/ and review the diff -- the "
        "TypeScript surfaces are compared against these files"
    )


# The body matcher is what makes a json_body route prove an argument reached a write, so
# it has to be shown to refuse things, not just to let the goldens through.
@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        (1, 1, True),
        (1, 1.0, True),  # one JSON number, as JavaScript sees it
        (True, 1, False),
        (1, True, False),
        (False, 0, False),
        (None, 0, False),
        ({"a": [1, {"b": True}]}, {"a": [1, {"b": True}]}, True),
        ({"a": [1, {"b": True}]}, {"a": [1, {"b": 1}]}, False),
        ({"a": 1}, {"a": 1, "b": 2}, False),
        ([1, 2], [1], False),
    ],
)
def test_same_json_compares_the_way_the_ts_replay_does(a: Any, b: Any, same: bool) -> None:
    assert _same_json(a, b) is same


@pytest.mark.parametrize(
    ("body", "matches"),
    [
        (b'{"version": 1, "inputs": {}}', True),
        (b'{"inputs": {}}', False),
        (b'{"version": 2}', False),
        (b'{"version": true}', False),
        (b"not json", False),
        (b"[1]", False),
        (None, False),
    ],
)
def test_json_body_matcher_refuses_a_body_without_the_value(
    body: bytes | None, matches: bool
) -> None:
    ok, _reason = _json_body_matcher({"version": 1})(SimpleNamespace(body=body))
    assert ok is matches


@pytest.mark.parametrize("version", [None, True])
def test_a_json_body_route_does_not_answer_a_post_without_the_value(
    version: bool | None,
) -> None:
    # with_version's table, called without the version and with a bool standing in for
    # it: if the matcher were not wired into run_case, both would succeed.
    case = Case(
        tool="invoke_workflow",
        name="with_version_missing",
        note="",
        input={},
        call=lambda: invoke_workflow_fn("wf000001", history_id="h0000", version=version),
        routes=[
            route(
                "/api/workflows/wf000001/invocations",
                {"id": "inv00002"},
                method="POST",
                json_body={"version": 1},
            )
        ],
    )
    with pytest.raises(ValueError, match="Invoke workflow failed"):
        run_case(case)
