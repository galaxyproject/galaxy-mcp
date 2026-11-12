"""Golden cases for a scoped session, added to the envelope fixtures.

Each case carries the scope it runs under, as the environment a host would set, and --
because a case is one call -- whatever an earlier call of the session would have left
behind: the histories it created and the ids it saw. What the call leaves behind in turn
is written into the index (``scopeAfter``), so the other server is held to remembering
the same things as well as to answering the same way.

Ids are hex here, unlike the rest of the fixtures: the scope's settings accept encoded
ids only, and a read by job or invocation id refuses anything else before asking Galaxy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .test_helpers import (
    create_history_fn,
    get_collection_details_fn,
    get_dataset_details_fn,
    get_histories_fn,
    get_history_contents_fn,
    get_job_details_fn,
    get_server_info_fn,
    get_user_fn,
    list_history_ids_fn,
    list_jobs_fn,
    run_tool_fn,
    update_history_fn,
)

A, B, C, D = "aaaa0001", "bbbb0002", "cccc0003", "dddd0004"
ENV = {"GALAXY_MCP_SCOPE_WRITE_HISTORIES": A, "GALAXY_MCP_SCOPE_READ_HISTORIES": B}

# (tool, name, note, input, call or None for a failure, routes, scope)
ScopedCase = tuple[
    str, str, str, dict[str, Any], Callable[[], Any] | None, list[dict[str, Any]], dict[str, Any]
]


def scope_cases(route: Callable[..., dict[str, Any]], version_route: dict[str, Any]) -> list[ScopedCase]:
    out: list[ScopedCase] = []

    def ok(tool, name, note, inp, call, routes, **scope):  # type: ignore[no-untyped-def]
        out.append((tool, name, note, inp, call, routes, {"env": ENV, **scope}))

    def refused(tool, name, note, inp, routes=(), **scope):  # type: ignore[no-untyped-def]
        out.append((tool, name, note, inp, None, list(routes), {"env": ENV, **scope}))

    histories = [
        {"id": A, "name": "assigned"},
        {"id": B, "name": "reference"},
        {"id": C, "name": "someone else's"},
    ]
    submission = {
        "outputs": [{"id": "f001", "hid": 1, "name": "out", "state": "new", "history_id": D}],
        "output_collections": [{"id": "c0f1"}],
        "jobs": [{"id": "e001", "state": "new", "tool_id": "cat1", "tool_version": "1.0.0"}],
        "implicit_collections": [],
    }

    ok(
        "get_server_info",
        "scoped",
        "a scoped session is told what it may touch, in data.scope and in the message",
        {},
        lambda: get_server_info_fn(),
        [route("/api/configuration", {"brand": "Scoped Galaxy"}), version_route],
    )
    ok(
        "get_server_info",
        "scoped_with_a_created_history",
        "a history the session created is listed as writable, and as the one it may delete",
        {},
        lambda: get_server_info_fn(),
        [route("/api/configuration", {"brand": "Scoped Galaxy"}), version_route],
        created=[D],
    )
    refused(
        "get_history_details",
        "history_outside_the_scope",
        "a history the scope does not list is refused before anything is asked of Galaxy",
        {"history_id": C},
    )
    refused(
        "run_tool",
        "read_only_history",
        "a write to a history the session may only read",
        {"history_id": B, "tool_id": "cat1", "inputs": {}},
    )
    refused(
        "get_dataset_details",
        "dataset_in_another_history",
        "the dataset's own record puts it in a history outside the scope, so it is not returned",
        {"dataset_id": "dd01"},
        [route("/api/datasets/dd01", {"id": "dd01", "name": "x", "state": "new", "history_id": C})],
    )
    ok(
        "get_dataset_details",
        "dataset_in_a_read_only_history",
        "a dataset in a history the session may read is returned, and what it names remembered",
        {"dataset_id": "dd02"},
        lambda: get_dataset_details_fn("dd02"),
        [
            route(
                "/api/datasets/dd02",
                {"id": "dd02", "name": "x", "state": "new", "history_id": B, "creating_job": "e002"},
            )
        ],
    )
    ok(
        "get_histories",
        "narrowed_to_the_scope",
        "only the session's histories are listed, and the count is theirs",
        {},
        lambda: get_histories_fn(),
        [route("/api/histories", histories)],
    )
    ok(
        "get_histories",
        "narrowed_then_paged",
        "narrowed before the window is cut, so the page and its total describe the same set",
        {"limit": 1, "offset": 1},
        lambda: get_histories_fn(limit=1, offset=1),
        [route("/api/histories", histories)],
    )
    ok(
        "list_history_ids",
        "narrowed_to_the_scope",
        "the compact listing is narrowed the same way",
        {},
        lambda: list_history_ids_fn(),
        [route("/api/histories", histories)],
    )
    refused(
        "create_history",
        "blocked_by_name",
        "a tool the host blocked is refused by name",
        {"history_name": "scratch"},
        env={**ENV, "GALAXY_MCP_SCOPE_BLOCKED_TOOLS": "create_history"},
    )
    refused(
        "create_history",
        "creating_not_allowed",
        "creating turned off refuses the tool that creates",
        {"history_name": "scratch"},
        env={**ENV, "GALAXY_MCP_SCOPE_ALLOW_CREATE": "0"},
    )
    ok(
        "create_history",
        "joins_the_scope",
        "a history the session creates is remembered as created",
        {"history_name": "scratch"},
        lambda: create_history_fn("scratch"),
        [route("/api/histories", {"id": D, "name": "scratch"}, method="POST")],
    )
    ok(
        "run_tool",
        "in_a_created_history",
        "a write to a history the session created earlier",
        {"history_id": D, "tool_id": "cat1", "inputs": {}},
        lambda: run_tool_fn(D, "cat1", {}),
        [route("/api/tools", submission, method="POST")],
        created=[D],
    )
    refused(
        "update_history",
        "delete_of_a_history_it_did_not_create",
        "writable is not deletable: only a history the session created may be deleted",
        {"history_id": A, "deleted": True},
        created=[D],
    )
    ok(
        "update_history",
        "delete_of_a_created_history",
        "the session may delete a history it created",
        {"history_id": D, "deleted": True},
        lambda: update_history_fn(D, deleted=True),
        [route(f"/api/histories/{D}", {"id": D, "name": "scratch", "deleted": True}, method="PUT")],
        created=[D],
    )
    ok(
        "update_history",
        "delete_of_a_writable_history_when_the_host_allows",
        "with the destructive rule set to writable, a writable history may be deleted",
        {"history_id": A, "deleted": True},
        lambda: update_history_fn(A, deleted=True),
        [route(f"/api/histories/{A}", {"id": A, "name": "assigned", "deleted": True}, method="PUT")],
        env={**ENV, "GALAXY_MCP_SCOPE_DESTRUCTIVE": "writable"},
    )
    refused(
        "get_dataset_details",
        "job_id_passed_as_a_dataset_id",
        "an id an earlier call returned as a job is refused where a dataset id belongs",
        {"dataset_id": "e001"},
        handles={"e001": ["job"]},
    )
    refused(
        "cancel_workflow_invocation",
        "invocation_in_another_history",
        "the invocation is read first, and the cancel never sent for another history's",
        {"invocation_id": "e00f"},
        [route("/api/invocations/e00f", {"id": "e00f", "history_id": C, "state": "scheduled"})],
    )
    ok(
        "get_user",
        "scoped",
        "a scoped session reads its id and username and nothing else of the account",
        {},
        lambda: get_user_fn(),
        [
            route(
                "/api/users/current",
                {
                    "id": "f00d",
                    "username": "curator",
                    "email": "curator@galaxy.example",
                    "preferences": {"theme": "dark"},
                    "quota_percent": 12,
                },
            )
        ],
    )
    ok(
        "list_jobs",
        "narrowed_to_the_scope",
        "jobs from histories outside the scope are dropped from the page",
        {},
        lambda: list_jobs_fn(),
        [
            route(
                "/api/jobs",
                [
                    {"id": "e001", "state": "ok", "tool_id": "cat1", "history_id": A},
                    {"id": "e002", "state": "ok", "tool_id": "cat1", "history_id": C},
                    {"id": "e003", "state": "ok", "tool_id": "cat1", "history_id": B},
                ],
            )
        ],
    )
    ok(
        "get_history_contents",
        "read_only_history",
        "a read of a read-only history, whose items are remembered by kind",
        {"history_id": B},
        lambda: get_history_contents_fn(B),
        [
            route(
                f"/api/histories/{B}/contents",
                {
                    "contents": [
                        {"id": "dd03", "name": "reads", "history_content_type": "dataset", "hid": 1},
                        {
                            "id": "c0c1",
                            "name": "pairs",
                            "history_content_type": "dataset_collection",
                            "hid": 2,
                        },
                    ],
                    "stats": {"total_matches": 2},
                },
            )
        ],
    )
    ok(
        "get_collection_details",
        "scoped",
        "a collection in scope, whose elements and the datasets behind them are remembered apart",
        {"collection_id": "c0c1"},
        lambda: get_collection_details_fn("c0c1"),
        [
            route(
                "/api/dataset_collections/c0c1",
                {
                    "id": "c0c1",
                    "name": "pairs",
                    "collection_type": "list",
                    "element_count": 1,
                    "populated": True,
                    "populated_state": "ok",
                    "history_id": A,
                    "elements": [
                        {
                            "id": "dce1",
                            "element_index": 0,
                            "element_identifier": "s1",
                            "element_type": "hda",
                            "object": {"id": "dd04", "name": "s1", "state": "ok", "file_ext": "txt"},
                        }
                    ],
                },
            )
        ],
    )
    refused(
        "get_collection_details",
        "collection_in_another_history",
        "a collection whose record names a history outside the scope is not returned",
        {"collection_id": "c0c2"},
        [
            route(
                "/api/dataset_collections/c0c2",
                {"id": "c0c2", "name": "theirs", "history_id": C, "elements": []},
            )
        ],
    )
    ok(
        "get_job_details",
        "scoped_by_job_id",
        "a job in scope, whose inputs and outputs are remembered",
        {"job_id": "e001"},
        lambda: get_job_details_fn(job_id="e001"),
        [
            route(
                "/api/jobs/e001",
                {
                    "id": "e001",
                    "state": "ok",
                    "tool_id": "cat1",
                    "history_id": A,
                    "inputs": {"input1": {"id": "dd05", "src": "hda"}},
                    "outputs": {"out1": {"id": "dd06", "src": "hda"}},
                },
            )
        ],
    )
    return out
