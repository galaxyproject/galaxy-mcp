"""An optional per-session scope: which histories a session may touch, enforced in one place.

A host that runs an agent -- an MCP client launching this server for one task -- sets
``GALAXY_MCP_SCOPE_*`` in the server's environment. Every tool is then wrapped, where it
is registered, by ``guard``: a history argument outside the scope and a blocked tool are
refused before any request; a record that belongs to another history is refused rather
than returned; listings are narrowed; ``get_user`` answers with id and username only;
``get_server_info`` reports the scope. With none of the variables set, ``guard`` hands the
call straight through and nothing here runs.

This is the Python half of one contract. ``TOOLS`` below is the classification of every
tool -- which parameter carries which kind of id, and what a scoped session does with the
tool beyond that -- and ``contract/scope.json`` is generated from it
(``uv run python -m tests.scope_manifest``); the TypeScript packages are tested against
that file, so the two servers cannot classify a tool differently. The sentences are the
TypeScript server's, byte for byte (``typescript/packages/galaxy-ops/src/scope.ts``), and
the shared goldens under ``contract/envelopes`` pin them on both sides: change both.

A wrapper at registration rather than FastMCP middleware, for two reasons. Middleware
sees a call only when it arrives over MCP, and the tools are also called as functions --
by the golden generator, by code mode, by tests -- which would then run unscoped. And
narrowing a result in middleware means rewriting a serialised ``ToolResult`` (its text
block and its structured content, in step), where the wrapper edits the ``GalaxyResult``
before anything is serialised.
"""

from __future__ import annotations

import functools
import inspect
import re
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCOPE_ENV = {
    "write": "GALAXY_MCP_SCOPE_WRITE_HISTORIES",
    "read": "GALAXY_MCP_SCOPE_READ_HISTORIES",
    "createdOnly": "GALAXY_MCP_SCOPE_CREATED_ONLY",
    "allowCreate": "GALAXY_MCP_SCOPE_ALLOW_CREATE",
    "destructive": "GALAXY_MCP_SCOPE_DESTRUCTIVE",
    "blocked": "GALAXY_MCP_SCOPE_BLOCKED_TOOLS",
}

_ENCODED_ID = re.compile(r"^[0-9a-fA-F]+$")

HANDLE_NOUNS: dict[str, str] = {
    "history": "history id",
    "hda": "dataset id",
    "hdca": "dataset collection id",
    "job": "job id",
    "invocation": "invocation id",
    "tool_request": "tool request id",
    "workflow": "workflow id",
    "dataset_collection_element": "collection element id",
}

DEFAULT_HANDLE_CAPACITY = 10_000


class ScopeConfigError(ValueError):
    """A scope setting that cannot be read. The server must not start unscoped instead."""


@dataclass(frozen=True)
class Scope:
    history_ids: tuple[str, ...] = ()
    read_only_history_ids: tuple[str, ...] = ()
    blocked_operations: tuple[str, ...] = ()
    include_created: bool = True
    allow_create: bool = True
    destructive: Literal["created", "writable"] = "created"


def _items(value: str | None) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def _flag(name: str, value: str | None) -> bool | None:
    if value is None or not value.strip():
        return None
    v = value.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    raise ScopeConfigError(f"{name} must be 1 or 0 (got '{value}')")


def _history_ids(name: str, value: str | None) -> list[str]:
    ids = _items(value)
    for item in ids:
        if not _ENCODED_ID.match(item):
            raise ScopeConfigError(
                f"{name} must be a comma-separated list of encoded history ids; '{item}' is not one"
            )
    return ids


def scope_from_env(
    env: Mapping[str, str | None], known_operations: Iterable[str] | None = None
) -> Scope | None:
    """The scope these settings describe, or None when none of them is set."""
    names = SCOPE_ENV
    present = [name for name in names.values() if (env.get(name) or "").strip()]
    if not present:
        return None

    write = _history_ids(names["write"], env.get(names["write"]))
    read = _history_ids(names["read"], env.get(names["read"]))
    created_only = _flag(names["createdOnly"], env.get(names["createdOnly"])) is True
    allow_create = _flag(names["allowCreate"], env.get(names["allowCreate"])) is not False
    raw_destructive = env.get(names["destructive"])
    destructive = (raw_destructive or "").strip().lower() or "created"
    if destructive not in {"created", "writable"}:
        raise ScopeConfigError(
            f"{names['destructive']} must be 'created' or 'writable' (got '{raw_destructive}')"
        )
    if created_only and write:
        raise ScopeConfigError(
            f"{names['createdOnly']}=1 means no pre-existing history may be written; "
            f"unset {names['write']} or it"
        )
    if created_only and not allow_create:
        raise ScopeConfigError(
            f"{names['createdOnly']}=1 with {names['allowCreate']}=0 leaves no history "
            "the session could ever write"
        )
    if not created_only and not write and not read:
        raise ScopeConfigError(
            f"a scope needs histories: set {names['write']} and/or {names['read']}, or "
            f"{names['createdOnly']}=1 (only {', '.join(present)} set)"
        )
    blocked = _items(env.get(names["blocked"]))
    if known_operations is not None:
        known = set(known_operations)
        unknown = [name for name in blocked if name not in known]
        if unknown:
            raise ScopeConfigError(f"{names['blocked']} names unknown tools: {', '.join(unknown)}")
    return Scope(
        history_ids=tuple(write),
        read_only_history_ids=tuple(read),
        blocked_operations=tuple(blocked),
        include_created=True,
        allow_create=allow_create,
        destructive=cast(Literal["created", "writable"], destructive),
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


@dataclass
class ScopeState:
    """One process's scope and what the session has done under it.

    Per process, like the configuration it comes from: a stdio server is one session, and
    the HTTP transports share it across their clients -- "created by the session" there
    means created since the server started.
    """

    scope: Scope
    created: list[str] = field(default_factory=list)
    # Each id's kinds in the order they were first seen, which is the order a refusal
    # names them in -- on both servers.
    handles: OrderedDict[str, list[str]] = field(default_factory=OrderedDict)
    capacity: int = DEFAULT_HANDLE_CAPACITY

    def created_histories(self) -> list[str]:
        return list(self.created) if self.scope.include_created else []

    def record_created(self, history_id: Any) -> None:
        if isinstance(history_id, str) and history_id and history_id.lower() not in self.created:
            self.created.append(history_id.lower())

    def record_handles(self, kind: str, ids: Iterable[Any]) -> None:
        for raw in ids:
            if not isinstance(raw, str) or not raw:
                continue
            key = raw.lower()
            seen = self.handles.get(key)
            if seen is not None:
                self.handles.move_to_end(key)
                if kind not in seen:
                    seen.append(kind)
                continue
            while len(self.handles) >= self.capacity:
                self.handles.popitem(last=False)
            self.handles[key] = [kind]

    def record_refs(self, refs: Iterable[Any]) -> None:
        kinds = {
            "hda": "hda",
            "hdca": "hdca",
            "dce": "dataset_collection_element",
            "job": "job",
        }
        for ref in refs:
            if isinstance(ref, dict) and isinstance(ref.get("src"), str):
                kind = kinds.get(ref["src"])
                if kind:
                    self.record_handles(kind, [ref.get("id")])

    def kinds_of(self, value: str) -> list[str] | None:
        return self.handles.get(value.lower()) or None


_state: ScopeState | None = None
# What a tool learned about the record it read that its envelope does not carry.
_noted: ContextVar[dict[str, Any] | None] = ContextVar("galaxy_mcp_scope_noted", default=None)


# Set when the scope's settings could not be read. The command line refuses to start on
# it; a process that imported the server some other way refuses every tool instead, so an
# unreadable scope is never an unscoped server.
config_error: str | None = None


def configure(scope: Scope | None) -> ScopeState | None:
    """Install the process's scope (or none). Returns the new state."""
    global _state
    _state = ScopeState(scope) if scope is not None else None
    return _state


def configure_from_env(env: Mapping[str, str | None]) -> None:
    """Read the process's scope from its environment, failing closed on a bad setting."""
    global config_error
    try:
        configure(scope_from_env(env, TOOLS))
    except ScopeConfigError as exc:
        config_error = str(exc)


def current() -> ScopeState | None:
    return _state


def restore(state: ScopeState | None) -> None:
    global _state
    _state = state


# ---------------------------------------------------------------------------
# Decisions and sentences (typescript/packages/galaxy-ops/src/scope.ts, byte for byte)
# ---------------------------------------------------------------------------


def _same(a: str, b: str) -> bool:
    return a.lower() == b.lower()


def allowed_histories(scope: Scope, access: str, created: list[str]) -> list[str]:
    writable = list(scope.history_ids) + [
        c for c in created if not any(_same(h, c) for h in scope.history_ids)
    ]
    return writable if access == "write" else writable + list(scope.read_only_history_ids)


def history_allowed(scope: Scope, history_id: str, access: str, created: list[str]) -> bool:
    return any(_same(a, history_id) for a in allowed_histories(scope, access, created))


def create_allowed(scope: Scope) -> bool:
    return scope.allow_create and "create_history" not in scope.blocked_operations


def history_deletable(scope: Scope, history_id: str, created: list[str]) -> bool:
    if scope.destructive == "writable":
        return history_allowed(scope, history_id, "write", created)
    return any(_same(c, history_id) for c in created)


def _list(ids: Iterable[str]) -> str:
    ids = list(ids)
    return ", ".join(ids) if ids else "none"


def _allowance(scope: Scope, access: str, created: list[str]) -> str:
    if access == "read":
        return f"Histories this run may read: {_list(allowed_histories(scope, 'read', created))}"
    read_only = scope.read_only_history_ids
    return f"Histories this run may write: {_list(allowed_histories(scope, 'write', created))}" + (
        f"; read-only: {_list(read_only)}" if read_only else ""
    )


def blocked_sentence(tool: str) -> str:
    return (
        f"{tool} is not available in this run: the host's scope blocks it. "
        "Nothing was sent to Galaxy."
    )


def history_out_of_scope_sentence(
    tool: str, param: str, history_id: str, scope: Scope, access: str, created: list[str]
) -> str:
    read_only_target = access == "write" and history_allowed(scope, history_id, "read", created)
    why = (
        f"{param} '{history_id}' is read-only in this run and {tool} changes it"
        if read_only_target
        else f"{param} '{history_id}' is outside this run's scope"
    )
    return f"{why}. {_allowance(scope, access, created)}. Nothing was sent to Galaxy."


def missing_param_sentence(tool: str, params: list[str], scope: Scope, created: list[str]) -> str:
    return (
        f"{tool} needs {' or '.join(params)} in this run: without it Galaxy would use its "
        "current history, which is not known to be one this run may write. "
        f"{_allowance(scope, 'write', created)}. Nothing was sent to Galaxy."
    )


def create_not_allowed_sentence(tool: str) -> str:
    return (
        f"{tool} would create a history, and this run may not create histories. "
        "Nothing was sent to Galaxy."
    )


def not_deletable_sentence(
    tool: str, param: str, history_id: str, scope: Scope, created: list[str]
) -> str:
    rule = (
        f"Histories this run may delete: {_list(allowed_histories(scope, 'write', created))}"
        if scope.destructive == "writable"
        else f"This run may delete only histories it created itself: {_list(created)}"
    )
    return (
        f"{tool} would delete {param} '{history_id}', which this run may not delete. "
        f"{rule}. Nothing was sent to Galaxy."
    )


def owner_out_of_scope_sentence(
    tool: str,
    noun: str,
    record_id: str,
    owner: str,
    scope: Scope,
    access: str,
    when: str,
    created: list[str],
) -> str:
    read_only_owner = access == "write" and history_allowed(scope, owner, "read", created)
    why = (
        f"belongs to history '{owner}', which is read-only in this run, and {tool} changes it"
        if read_only_owner
        else f"belongs to history '{owner}', which is outside this run's scope"
    )
    outcome = f"{tool} was not run." if when == "before" else "It was not returned."
    return f"{noun} '{record_id}' {why}. {_allowance(scope, access, created)}. {outcome}"


def kind_mismatch_sentence(param: str, value: str, seen: Iterable[str], expected: str) -> str:
    nouns = [f"a {HANDLE_NOUNS[k]}" for k in seen]
    described = nouns[0] if len(nouns) == 1 else f"{', '.join(nouns[:-1])} and {nouns[-1]}"
    return (
        f"{param} '{value}' was returned by an earlier call as {described}; "
        f"{param} expects a {HANDLE_NOUNS[expected]}. Nothing was sent to Galaxy."
    )


def scope_summary(state: ScopeState) -> dict[str, Any]:
    scope = state.scope
    created = state.created_histories()
    writable = allowed_histories(scope, "write", created)
    readable = allowed_histories(scope, "read", created)
    create = create_allowed(scope)
    if create:
        creating = (
            "It may create histories, and use the ones it creates. "
            if scope.include_created
            else "It may create histories but not then use them. "
        )
    else:
        creating = "It may not create histories. "
    note = (
        f"This session is scoped. It may read histories: {_list(readable)}; "
        f"write to: {_list(writable)}. "
        + creating
        + (
            "It may delete any history it may write."
            if scope.destructive == "writable"
            else "It may delete only histories it created itself."
        )
    )
    return {
        "readable_history_ids": readable,
        "writable_history_ids": writable,
        "created_history_ids": created,
        "create_allowed": create,
        "destructive": scope.destructive,
        "blocked_operations": list(scope.blocked_operations),
        "note": note,
    }


# ---------------------------------------------------------------------------
# The classification. contract/scope.json is generated from this table.
# ---------------------------------------------------------------------------

# Wire parameter names that always carry an id of a known kind. A tool that takes one and
# does not classify it cannot be registered -- see ``guard``.
KNOWN_ID_PARAMS: dict[str, str] = {
    "history_id": "history",
    "dataset_id": "hda",
    "collection_id": "hdca",
    "job_id": "job",
    "invocation_id": "invocation",
    "workflow_id": "workflow",
    "tool_request_id": "tool_request",
}

_HISTORY = {"history_id": "history"}

# Keys: ids (parameter -> kind, in the order a refusal names them), access, requires (each
# entry satisfied by any one of its names), and the named strategies implemented below --
# creates, destroys, ownerOf (before the call, one request), owner (after it), filter,
# restrict, annotate. ``records`` is how this server remembers the ids a result carried.
TOOLS: dict[str, dict[str, Any]] = {
    "cancel_workflow_invocation": {
        "ids": {"invocation_id": "invocation"},
        "access": "write",
        "ownerOf": "invocation",
        "owner": "cancelled_invocation",
        "records": ["cancelled_invocation"],
    },
    "connect": {"access": "write", "only": "python"},
    "create_history": {"access": "write", "creates": "always", "records": ["created_history"]},
    "create_page": {"ids": _HISTORY, "access": "write"},
    "create_user_tool": {"access": "write"},
    "delete_user_tool": {"access": "write"},
    "download_dataset": {"ids": {"dataset_id": "hda"}, "access": "read", "ownerOf": "dataset"},
    "get_collection_details": {
        "ids": {"collection_id": "hdca"},
        "access": "read",
        "owner": "collection",
        "records": ["collection"],
    },
    "get_dataset_details": {
        "ids": {"dataset_id": "hda"},
        "access": "read",
        "owner": "dataset",
        "records": ["dataset"],
    },
    "get_histories": {"access": "read", "filter": "histories_in_tool"},
    "get_history_contents": {"ids": _HISTORY, "access": "read", "records": ["contents"]},
    "get_history_details": {"ids": _HISTORY, "access": "read", "records": ["history"]},
    "get_invocations": {
        "ids": {
            "invocation_id": "invocation",
            "workflow_id": "workflow",
            "history_id": "history",
        },
        "access": "read",
        "owner": "invocation",
        "filter": "invocation_rows",
        "records": ["invocations"],
    },
    "get_iwc_workflow_details": {"access": "read"},
    "get_iwc_workflows": {"access": "read"},
    "get_job_details": {
        "ids": {"dataset_id": "hda", "history_id": "history", "job_id": "job"},
        "access": "read",
        "owner": "job",
        "records": ["job"],
    },
    "get_job_logs": {"ids": {"job_id": "job"}, "access": "read", "ownerOf": "job"},
    "get_page": {"access": "read"},
    "get_page_revision": {"access": "read"},
    "get_server_info": {"access": "read", "annotate": "server_info"},
    "get_tool_citations": {"access": "read"},
    "get_tool_details": {"access": "read"},
    "get_tool_input_template": {"access": "read"},
    "get_tool_panel": {"access": "read"},
    "get_tool_run_examples": {"access": "read"},
    "get_user": {"access": "read", "restrict": "user"},
    "get_user_tool": {"access": "read"},
    "get_workflow_details": {
        "ids": {"workflow_id": "workflow"},
        "access": "read",
        "records": ["workflow"],
    },
    "get_workflow_input_template": {
        "ids": {"workflow_id": "workflow", "history_id": "history"},
        "access": "read",
    },
    "import_workflow_from_iwc": {"access": "write", "records": ["workflow"]},
    "invoke_workflow": {
        "ids": {"workflow_id": "workflow", "history_id": "history"},
        "access": "write",
        "requires": [["history_id", "history_name"]],
        "creates": "history_name_without_history_id",
        "records": ["invoked"],
    },
    "list_history_ids": {"access": "read", "filter": "histories_in_tool"},
    "list_jobs": {
        "ids": _HISTORY,
        "access": "read",
        "filter": "job_rows",
        "records": ["job_rows"],
    },
    "list_page_revisions": {"access": "read"},
    "list_pages": {"ids": _HISTORY, "access": "read"},
    "list_user_tools": {"access": "read"},
    "list_workflows": {"access": "read", "records": ["workflow_rows"]},
    "recommend_biocontainer": {"access": "read"},
    "recommend_iwc_workflows": {"access": "read"},
    "revert_page_revision": {"access": "write"},
    "run_tool": {"ids": _HISTORY, "access": "write", "records": ["submission"]},
    "run_user_tool": {"ids": _HISTORY, "access": "write", "records": ["submission"]},
    "search_iwc_workflows": {"access": "read"},
    "search_tools_by_keywords": {"access": "read"},
    "search_tools_by_name": {"access": "read"},
    "update_history": {"ids": _HISTORY, "access": "write", "destroys": "deleted_true"},
    "update_page": {"access": "write"},
    "upload_file": {
        "ids": _HISTORY,
        "access": "write",
        "requires": [["history_id"]],
        "records": ["submission", "history_argument"],
    },
    "upload_file_from_url": {
        "ids": _HISTORY,
        "access": "write",
        "requires": [["history_id"]],
        "records": ["submission"],
    },
}


class _RefusedError(ValueError):
    """A refusal of the scope's own. A ValueError, so FastMCP words it like any other."""


def _id_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str) and v]
    return []


def _record_noun(spec: dict[str, Any], args: Mapping[str, Any]) -> tuple[str, str]:
    for param, kind in spec.get("ids", {}).items():
        if kind == "history":
            continue
        values = _id_values(args.get(param))
        if values:
            return HANDLE_NOUNS[kind].removesuffix(" id"), values[0]
    return "record", ""


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _rows(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _history_of(record: Any) -> str | None:
    owner = _dict(record).get("history_id")
    return owner if isinstance(owner, str) and owner else None


# -- strategies: creates / destroys ------------------------------------------------------

_CREATES: dict[str, Callable[[Mapping[str, Any]], bool]] = {
    "always": lambda args: True,
    "history_name_without_history_id": lambda args: (
        not args.get("history_id") and bool(args.get("history_name"))
    ),
}
_DESTROYS: dict[str, Callable[[Mapping[str, Any]], bool]] = {
    "deleted_true": lambda args: args.get("deleted") is True,
}


# -- strategies: ownerOf (one request before the call) -----------------------------------


def _lookup_owner(kind: str, args: Mapping[str, Any], gi: Any) -> str | None:
    if kind == "invocation":
        value = args.get("invocation_id")
        if not isinstance(value, str) or not _ENCODED_ID.match(value):
            return None
        return _history_of(gi.invocations.show_invocation(value))
    if kind == "dataset":
        return _history_of(gi.datasets.show_dataset(args.get("dataset_id")))
    if kind == "job":
        value = args.get("job_id")
        if not isinstance(value, str) or not _ENCODED_ID.match(value):
            return None
        record = gi.jobs.show_job(value)
        if _dict(record).get("id", "").lower() != value.lower():
            return None
        return _history_of(record)
    raise KeyError(kind)


# -- strategies: owner (read off the result) ---------------------------------------------


def _owner_after(kind: str, data: Any) -> str | None:
    if kind == "dataset":
        return _history_of(_dict(data).get("dataset"))
    if kind == "job":
        return _history_of(_dict(data).get("job"))
    if kind == "invocation":
        return _history_of(data)  # a listing is a list, which has no owner
    if kind == "cancelled_invocation":
        return _history_of(_dict(data).get("invocation"))
    if kind == "collection":
        noted = _noted.get() or {}
        owner = noted.get("owner")
        return owner if isinstance(owner, str) and owner else None
    raise KeyError(kind)


# -- strategies: records -----------------------------------------------------------------


def _ids(rows: Any, key: str = "id") -> list[Any]:
    return [_dict(row).get(key) for row in rows] if isinstance(rows, list) else []


def _record(  # noqa: PLR0912 -- one flat table of strategies
    state: ScopeState, kind: str, data: Any, args: Mapping[str, Any]
) -> None:
    d = _dict(data)
    if kind == "created_history":
        state.record_handles("history", [d.get("id")])
        state.record_created(d.get("id"))
    elif kind == "history":
        state.record_handles("history", [_dict(d.get("history")).get("id")])
    elif kind == "contents":
        rows = _rows(d.get("contents"))
        for content_type, handle in (("dataset", "hda"), ("dataset_collection", "hdca")):
            state.record_handles(
                handle,
                [
                    _dict(r).get("id")
                    for r in rows
                    if _dict(r).get("history_content_type") == content_type
                ],
            )
    elif kind == "dataset":
        record = _dict(d.get("dataset"))
        state.record_handles("hda", [record.get("id")])
        state.record_handles("history", [record.get("history_id")])
        state.record_handles("job", [record.get("creating_job")])
    elif kind == "collection":
        noted = _noted.get() or {}
        state.record_handles("hdca", [_dict(d.get("collection")).get("id")])
        state.record_handles("history", [noted.get("owner")])
        state.record_handles("dataset_collection_element", noted.get("elements", []))
        elements = _rows(d.get("elements"))
        for element_type, handle in (("hda", "hda"), ("dataset_collection", "hdca")):
            state.record_handles(
                handle,
                [
                    _dict(e).get("object_id")
                    for e in elements
                    if _dict(e).get("element_type") == element_type
                ],
            )
    elif kind == "job":
        job = _dict(d.get("job"))
        state.record_handles("job", [job.get("id")])
        state.record_handles("history", [job.get("history_id")])
        for key in ("inputs", "outputs"):
            refs = job.get(key)
            if isinstance(refs, dict):
                state.record_refs(refs.values())
        if d.get("dataset_id"):
            state.record_handles("hda", [d.get("dataset_id")])
    elif kind == "job_rows":
        state.record_handles("job", _ids(data))
        state.record_handles("history", _ids(data, "history_id"))
    elif kind == "submission":
        state.record_handles("job", _ids(d.get("jobs")))
        state.record_handles("hda", _ids(d.get("outputs")))
        state.record_handles("hdca", _ids(d.get("output_collections")))
        state.record_handles("hdca", _ids(d.get("implicit_collections")))
    elif kind == "history_argument":
        state.record_handles("history", [args.get("history_id")])
    elif kind == "invocations":
        for inv in data if isinstance(data, list) else [data]:
            record = _dict(inv)
            state.record_handles("invocation", [record.get("id")])
            state.record_handles("history", [record.get("history_id")])
            state.record_handles("workflow", [record.get("workflow_id")])
            if isinstance(record.get("steps"), list):
                state.record_handles("job", _ids(record["steps"], "job_id"))
            for key in ("inputs", "outputs", "output_collections"):
                refs = record.get(key)
                if isinstance(refs, dict):
                    state.record_refs(refs.values())
                elif isinstance(refs, list):
                    state.record_refs(refs)
    elif kind == "invoked":
        by_name = not args.get("history_id") and bool(args.get("history_name"))
        for inv in data if isinstance(data, list) else [data]:
            record = _dict(inv)
            state.record_handles("invocation", [record.get("id")])
            state.record_handles("history", [record.get("history_id")])
            state.record_handles("workflow", [record.get("workflow_id")])
            if by_name:
                state.record_created(record.get("history_id"))
    elif kind == "cancelled_invocation":
        state.record_handles("invocation", [_dict(d.get("invocation")).get("id")])
    elif kind == "workflow":
        state.record_handles("workflow", [d.get("id")])
    elif kind == "workflow_rows":
        state.record_handles("workflow", _ids(data))
    else:
        raise KeyError(kind)


# -- helpers the few tools that must report a fact call ----------------------------------


def visible_histories(histories: Any) -> Any:
    """The histories a scoped session may see, of the ones Galaxy listed.

    Called by the two history listings on what Galaxy returned, before they page or count,
    so a scoped page and its total describe the same set. Unscoped, the list comes back
    as it was.
    """
    state = _state
    if state is None or not isinstance(histories, list):
        return histories
    created = state.created_histories()
    seen = [
        h
        for h in histories
        if isinstance(_dict(h).get("id"), str)
        and history_allowed(state.scope, h["id"], "read", created)
    ]
    state.record_handles("history", _ids(seen))
    return seen


def note_collection(record: Any, elements: Any) -> None:
    """What get_collection_details read that its envelope leaves out: the owner, the elements."""
    if _state is None:
        return
    _noted.set(
        {
            "owner": _dict(record).get("history_id"),
            "elements": _ids(elements),
        }
    )


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def _before(
    state: ScopeState,
    tool: str,
    spec: dict[str, Any],
    args: Mapping[str, Any],
    get_gi: Callable[[], Any],
) -> None:
    scope = state.scope
    access = spec["access"]
    if tool in scope.blocked_operations:
        raise _RefusedError(blocked_sentence(tool))
    created = state.created_histories()
    destroys = "destroys" in spec and _DESTROYS[spec["destroys"]](args)
    for param, kind in spec.get("ids", {}).items():
        for value in _id_values(args.get(param)):
            seen = state.kinds_of(value)
            if seen and kind not in seen:
                raise _RefusedError(kind_mismatch_sentence(param, value, seen, kind))
            if kind != "history":
                continue
            if not history_allowed(scope, value, access, created):
                raise _RefusedError(
                    history_out_of_scope_sentence(tool, param, value, scope, access, created)
                )
            if destroys and not history_deletable(scope, value, created):
                raise _RefusedError(not_deletable_sentence(tool, param, value, scope, created))
    for names in spec.get("requires", []):
        if not any(_id_values(args.get(name)) for name in names):
            raise _RefusedError(missing_param_sentence(tool, names, scope, created))
    if "creates" in spec and _CREATES[spec["creates"]](args) and not create_allowed(scope):
        raise _RefusedError(create_not_allowed_sentence(tool))
    if "ownerOf" in spec:
        try:
            owner = _lookup_owner(spec["ownerOf"], args, get_gi())
        except Exception:
            # The lookup could not say. The tool runs and fails in its own words.
            owner = None
        if owner and not history_allowed(scope, owner, access, created):
            noun, record_id = _record_noun(spec, args)
            raise _RefusedError(
                owner_out_of_scope_sentence(
                    tool, noun, record_id, owner, scope, access, "before", created
                )
            )


def _after(
    state: ScopeState, tool: str, spec: dict[str, Any], args: Mapping[str, Any], result: Any
) -> Any:
    if not hasattr(result, "data"):
        return result
    scope = state.scope
    access = spec["access"]
    for kind in spec.get("records", []):
        _record(state, kind, result.data, args)
    created = state.created_histories()
    if "owner" in spec:
        owner = _owner_after(spec["owner"], result.data)
        if owner and not history_allowed(scope, owner, access, created):
            noun, record_id = _record_noun(spec, args)
            raise _RefusedError(
                owner_out_of_scope_sentence(
                    tool, noun, record_id, owner, scope, access, "after", created
                )
            )
    narrowing = spec.get("filter")
    if narrowing in {"job_rows", "invocation_rows"} and isinstance(result.data, list):
        rows = [
            row
            for row in result.data
            if isinstance(_dict(row).get("history_id"), str)
            and history_allowed(scope, row["history_id"], "read", created)
        ]
        noun = "jobs" if narrowing == "job_rows" else "workflow invocations"
        result.data = rows
        result.count = len(rows)
        result.message = f"Retrieved {len(rows)} {noun}"
    if spec.get("restrict") == "user":
        user = _dict(result.data)
        result.data = {
            key: user[key] for key in ("id", "username") if isinstance(user.get(key), str)
        }
    if spec.get("annotate") == "server_info" and isinstance(result.data, dict):
        summary = scope_summary(state)
        result.data["scope"] = summary
        result.message = f"{result.message}. {summary['note']}"
    return result


def guard(tool: str, fn: F, get_gi: Callable[[], Any]) -> F:
    """Wrap a tool so a scoped process holds it to the scope. Unscoped, a pass-through.

    Refuses to wrap a tool this module has not classified, or one that takes a parameter
    spelled like a known id parameter without saying what it is: a new tool cannot slip
    past the scope by being forgotten.
    """
    spec = TOOLS.get(tool)
    if spec is None:
        raise RuntimeError(
            f'tool "{tool}" has no entry in galaxy_mcp.scope.TOOLS, so the scope cannot classify it'
        )
    signature = inspect.signature(fn)
    for param in signature.parameters:
        if param in KNOWN_ID_PARAMS and param not in spec.get("ids", {}):
            raise RuntimeError(
                f'tool "{tool}" takes "{param}" but galaxy_mcp.scope.TOOLS declares no id for it'
            )
    for param in list(spec.get("ids", {})) + [
        n for names in spec.get("requires", []) for n in names
    ]:
        if param not in signature.parameters:
            raise RuntimeError(
                f'galaxy_mcp.scope.TOOLS names "{param}", which "{tool}" does not take'
            )

    @functools.wraps(fn)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        if config_error is not None:
            raise ValueError(f"This server's scope is misconfigured: {config_error}")
        state = _state
        if state is None:
            return fn(*args, **kwargs)
        given = signature.bind_partial(*args, **kwargs).arguments
        _before(state, tool, spec, given, get_gi)
        token = _noted.set(None)
        try:
            return _after(state, tool, spec, given, fn(*args, **kwargs))
        finally:
            _noted.reset(token)

    return cast(F, guarded)
