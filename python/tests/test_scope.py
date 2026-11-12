"""The optional per-session scope: its settings, its classification, and what it refuses.

The answers a scoped tool gives -- sentence for sentence, and what the session remembers
afterwards -- are pinned by the shared goldens (``tests/scope_fixtures.py``), which the
TypeScript server replays. What is here is the rest: the settings, the table's own
consistency, the state a session builds up across calls, and that no scope means no change.
"""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

from galaxy_mcp import scope, server

from .scope_manifest import MANIFEST_PATH, REGENERATE_COMMAND, render
from .test_helpers import (
    create_history_fn,
    get_histories_fn,
    get_job_logs_fn,
    get_user_fn,
    invoke_workflow_fn,
    run_tool_fn,
    update_history_fn,
)

A, B, C, D = "aaaa0001", "bbbb0002", "cccc0003", "dddd0004"
ENV = {scope.SCOPE_ENV["write"]: A, scope.SCOPE_ENV["read"]: B}


@pytest.fixture
def scoped(mcp_server_instance):
    """A scoped session over the mocked Galaxy; the scope is taken off again afterwards."""
    previous = scope.current()
    state = scope.configure(scope.scope_from_env(ENV))
    try:
        yield state, server.galaxy_state["gi"]
    finally:
        scope.restore(previous)


class TestSettings:
    def test_nothing_set_is_no_scope(self):
        assert scope.scope_from_env({}) is None
        assert scope.scope_from_env({scope.SCOPE_ENV["write"]: "  "}) is None

    def test_reads_every_setting(self):
        env = {
            scope.SCOPE_ENV["write"]: "a1, b2",
            scope.SCOPE_ENV["read"]: "c3",
            scope.SCOPE_ENV["allowCreate"]: "0",
            scope.SCOPE_ENV["destructive"]: "writable",
            scope.SCOPE_ENV["blocked"]: "upload_file",
        }
        assert scope.scope_from_env(env, scope.TOOLS) == scope.Scope(
            history_ids=("a1", "b2"),
            read_only_history_ids=("c3",),
            blocked_operations=("upload_file",),
            allow_create=False,
            destructive="writable",
        )
        created_only = scope.scope_from_env({scope.SCOPE_ENV["createdOnly"]: "1"})
        assert created_only == scope.Scope()

    @pytest.mark.parametrize(
        ("env", "said"),
        [
            ({"GALAXY_MCP_SCOPE_WRITE_HISTORIES": "not-an-id"}, "'not-an-id' is not one"),
            (
                {"GALAXY_MCP_SCOPE_WRITE_HISTORIES": "a1", "GALAXY_MCP_SCOPE_DESTRUCTIVE": "all"},
                "GALAXY_MCP_SCOPE_DESTRUCTIVE must be 'created' or 'writable' (got 'all')",
            ),
            (
                {"GALAXY_MCP_SCOPE_WRITE_HISTORIES": "a1", "GALAXY_MCP_SCOPE_ALLOW_CREATE": "maybe"},
                "GALAXY_MCP_SCOPE_ALLOW_CREATE must be 1 or 0 (got 'maybe')",
            ),
            (
                {"GALAXY_MCP_SCOPE_WRITE_HISTORIES": "a1", "GALAXY_MCP_SCOPE_CREATED_ONLY": "1"},
                "GALAXY_MCP_SCOPE_CREATED_ONLY=1 means no pre-existing history may be written",
            ),
            (
                {"GALAXY_MCP_SCOPE_CREATED_ONLY": "1", "GALAXY_MCP_SCOPE_ALLOW_CREATE": "0"},
                "leaves no history the session could ever write",
            ),
            ({"GALAXY_MCP_SCOPE_DESTRUCTIVE": "writable"}, "a scope needs histories"),
            (
                {"GALAXY_MCP_SCOPE_WRITE_HISTORIES": "a1", "GALAXY_MCP_SCOPE_BLOCKED_TOOLS": "nope"},
                "GALAXY_MCP_SCOPE_BLOCKED_TOOLS names unknown tools: nope",
            ),
        ],
    )
    def test_an_unreadable_setting_fails_by_name(self, env, said):
        with pytest.raises(scope.ScopeConfigError) as refused:
            scope.scope_from_env(env, scope.TOOLS)
        assert said in str(refused.value)


class TestClassification:
    def test_the_checked_in_contract_is_current(self):
        assert MANIFEST_PATH.read_text() == render(), (
            f"contract/scope.json is stale; run `{REGENERATE_COMMAND}` from python/"
        )

    def test_every_registered_tool_is_classified_and_nothing_else(self):
        registered = {tool.name for tool in asyncio.run(server.mcp.list_tools())}
        # Registered only with the container-recommend extra, classified either way.
        assert registered | {"recommend_biocontainer"} == set(scope.TOOLS)

    def test_an_unclassified_tool_cannot_be_registered(self):
        def brand_new_tool(history_id: str) -> None: ...

        with pytest.raises(RuntimeError, match="has no entry in galaxy_mcp.scope.TOOLS"):
            scope.guard("brand_new_tool", brand_new_tool, lambda: None)

    def test_a_known_id_parameter_has_to_be_declared(self):
        def get_user(history_id: str) -> None: ...

        with pytest.raises(RuntimeError, match='takes "history_id" but'):
            scope.guard("get_user", get_user, lambda: None)

    def test_a_declared_parameter_has_to_exist(self):
        def get_history_details() -> None: ...

        with pytest.raises(RuntimeError, match='names "history_id"'):
            scope.guard("get_history_details", get_history_details, lambda: None)


class TestNoScope:
    def test_the_server_is_unscoped_unless_configured(self):
        assert scope.current() is None

    def test_tools_answer_as_they_always_did(self, mcp_server_instance):
        gi = server.galaxy_state["gi"]
        gi.histories.get_histories.return_value = [{"id": C, "name": "theirs"}]
        gi.users.get_current_user.return_value = {"id": "u1", "username": "x", "email": "x@y.z"}
        assert get_histories_fn().data == [{"id": C, "name": "theirs"}]
        assert get_user_fn().data == {"id": "u1", "username": "x", "email": "x@y.z"}
        assert scope.current() is None


class TestSession:
    def test_a_created_history_joins_the_scope_and_is_the_one_it_may_delete(self, scoped):
        state, gi = scoped
        gi.tools.run_tool.return_value = {"jobs": [{"id": "e001"}], "outputs": [{"id": "f001"}]}
        with pytest.raises(ValueError, match="history_id 'dddd0004' is outside this run's scope"):
            run_tool_fn(D, "cat1", {})

        gi.histories.create_history.return_value = {"id": D, "name": "scratch"}
        create_history_fn("scratch")
        assert state.created == [D]
        assert run_tool_fn(D, "cat1", {}).success

        gi.histories.get_histories.return_value = [{"id": A}, {"id": B}, {"id": C}, {"id": D}]
        assert [h["id"] for h in get_histories_fn().data] == [A, B, D]

        with pytest.raises(ValueError) as refused:
            update_history_fn(A, deleted=True)
        assert str(refused.value) == (
            f"update_history would delete history_id '{A}', which this run may not delete. "
            f"This run may delete only histories it created itself: {D}. "
            "Nothing was sent to Galaxy."
        )
        gi.histories.update_history.assert_not_called()
        gi.histories.update_history.return_value = {"id": A, "name": "renamed"}
        assert update_history_fn(A, name="renamed").success
        assert update_history_fn(D, deleted=True).success

    def test_a_workflow_run_by_name_creates_a_history_that_joins_too(self, scoped):
        state, gi = scoped
        with pytest.raises(ValueError, match="needs history_id or history_name in this run"):
            invoke_workflow_fn("ab12")
        gi.workflows.invoke_workflow.return_value = {
            "id": "e00f",
            "history_id": D,
            "workflow_id": "ab12",
        }
        assert invoke_workflow_fn("ab12", history_name="fresh").success
        assert state.created == [D]
        assert state.handles["e00f"] == ["invocation"]

    def test_job_logs_are_refused_before_they_are_read(self, scoped):
        _state, gi = scoped
        gi.jobs = Mock()
        gi.jobs.show_job.return_value = {"id": "e001", "history_id": C}
        with pytest.raises(ValueError) as refused:
            get_job_logs_fn("e001")
        assert str(refused.value) == (
            f"job 'e001' belongs to history '{C}', which is outside this run's scope. "
            f"Histories this run may read: {A}, {B}. get_job_logs was not run."
        )
        gi.jobs.show_job.assert_called_once_with("e001")

    def test_the_handle_memory_is_bounded_and_forgets_the_least_recent(self):
        state = scope.ScopeState(scope.Scope(history_ids=("a1",)), capacity=3)
        state.record_handles("hda", ["A", "b", "c"])
        state.record_handles("job", ["a"])
        state.record_handles("hda", ["d"])
        assert list(state.handles) == ["c", "a", "d"]
        assert state.kinds_of("A") == ["hda", "job"]
        assert state.kinds_of("b") is None

    def test_a_refusal_reaches_an_mcp_client_like_any_other_failure(self, scoped):
        from fastmcp import Client

        async def call():
            async with Client(server.mcp) as client:
                return await client.call_tool_mcp("get_history_details", {"history_id": C})

        result = asyncio.run(call())
        assert result.isError is True
        assert result.content[0].text == (
            f"Error calling tool 'get_history_details': history_id '{C}' is outside this "
            f"run's scope. Histories this run may read: {A}, {B}. Nothing was sent to Galaxy."
        )


class TestMisconfigured:
    def test_a_bad_setting_refuses_every_tool_rather_than_running_unscoped(
        self, mcp_server_instance, monkeypatch
    ):
        monkeypatch.setattr(scope, "config_error", None)
        scope.configure_from_env({scope.SCOPE_ENV["write"]: "not-an-id"})
        assert scope.config_error is not None
        assert scope.current() is None
        with pytest.raises(ValueError, match="scope is misconfigured: GALAXY_MCP_SCOPE_WRITE"):
            get_user_fn()

    def test_the_command_line_refuses_to_start(self, monkeypatch, capsys):
        from galaxy_mcp import __main__ as entry

        monkeypatch.setattr(scope, "config_error", "GALAXY_MCP_SCOPE_DESTRUCTIVE must be ...")
        monkeypatch.setattr("sys.argv", ["galaxy-mcp"])
        with pytest.raises(SystemExit) as stopped:
            entry.run()
        assert stopped.value.code == 2
        assert "GALAXY_MCP_SCOPE_DESTRUCTIVE must be" in capsys.readouterr().err
