"""Generate the checked-in classification both servers hold their scope to.

`contract/scope.json` says, for every tool, which parameter carries which kind of id and
what a scoped session does with the tool beyond that, plus the settings the scope is read
from and every refusal sentence rendered for one fixed set of arguments. It is written
from `galaxy_mcp.scope` -- the table the Python server enforces -- and the TypeScript
packages are tested against it, so the two servers cannot classify a tool differently or
word a refusal differently without one of the two suites failing.

Regenerate with `uv run python -m tests.scope_manifest`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from galaxy_mcp import scope

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "contract" / "scope.json"
REGENERATE_COMMAND = "uv run python -m tests.scope_manifest"

_SCOPE = scope.Scope(history_ids=("a1",), read_only_history_ids=("b2",))
_WRITABLE = scope.Scope(history_ids=("a1",), read_only_history_ids=("b2",), destructive="writable")
_CREATED = ["d4"]


def sentences() -> dict[str, str]:
    """Each refusal, rendered once. The other server renders the same calls and compares."""
    s = scope
    state = scope.ScopeState(_SCOPE, created=list(_CREATED))
    return {
        "blocked": s.blocked_sentence("create_history"),
        "history_out_of_scope_read": s.history_out_of_scope_sentence(
            "get_history_details", "history_id", "c3", _SCOPE, "read", _CREATED
        ),
        "history_out_of_scope_write": s.history_out_of_scope_sentence(
            "run_tool", "history_id", "c3", _SCOPE, "write", _CREATED
        ),
        "history_read_only": s.history_out_of_scope_sentence(
            "update_history", "history_id", "b2", _SCOPE, "write", _CREATED
        ),
        "missing_param": s.missing_param_sentence("upload_file", ["history_id"], _SCOPE, _CREATED),
        "missing_param_either": s.missing_param_sentence(
            "invoke_workflow", ["history_id", "history_name"], _SCOPE, []
        ),
        "create_not_allowed": s.create_not_allowed_sentence("create_history"),
        "not_deletable_created": s.not_deletable_sentence(
            "update_history", "history_id", "a1", _SCOPE, _CREATED
        ),
        "not_deletable_writable": s.not_deletable_sentence(
            "update_history", "history_id", "c3", _WRITABLE, _CREATED
        ),
        "owner_after": s.owner_out_of_scope_sentence(
            "get_dataset_details", "dataset", "dd", "c3", _SCOPE, "read", "after", _CREATED
        ),
        "owner_before": s.owner_out_of_scope_sentence(
            "cancel_workflow_invocation", "invocation", "ee", "c3", _SCOPE, "write", "before", []
        ),
        "owner_read_only": s.owner_out_of_scope_sentence(
            "cancel_workflow_invocation", "invocation", "ee", "b2", _SCOPE, "write", "before", []
        ),
        "kind_mismatch": s.kind_mismatch_sentence("dataset_id", "ee", ["job"], "hda"),
        "kind_mismatch_two": s.kind_mismatch_sentence(
            "collection_id", "ee", ["job", "hda"], "hdca"
        ),
        "summary_note": s.scope_summary(state)["note"],
    }


def build() -> dict[str, Any]:
    tools: dict[str, Any] = {}
    for name in sorted(scope.TOOLS):
        spec = scope.TOOLS[name]
        row: dict[str, Any] = {"ids": dict(spec.get("ids", {})), "access": spec["access"]}
        for key in (
            "requires",
            "creates",
            "destroys",
            "ownerOf",
            "owner",
            "filter",
            "restrict",
            "annotate",
            "only",
        ):
            if key in spec:
                row[key] = spec[key]
        tools[name] = row
    return {
        "$comment": (
            "Generated from python/src/galaxy_mcp/scope.py -- do not edit by hand. Regenerate "
            f"with `{REGENERATE_COMMAND}` in python/. `tools` is keyed by wire tool name and "
            "spelled in wire parameter names; `ids` keeps the order a refusal names records "
            "in; each `requires` entry is satisfied by any one of its names; the other keys "
            "name a strategy each server implements once (a TypeScript op declares the "
            "matching function). `only` marks a tool one server has and the other does not. "
            "`sentences` is every refusal rendered for one fixed set of arguments."
        ),
        "env": dict(scope.SCOPE_ENV),
        "idParams": dict(scope.KNOWN_ID_PARAMS),
        "kinds": dict(scope.HANDLE_NOUNS),
        "handleCapacity": scope.DEFAULT_HANDLE_CAPACITY,
        "tools": tools,
        "sentences": sentences(),
    }


def render() -> str:
    return json.dumps(build(), indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    MANIFEST_PATH.write_text(render(), newline="\n")
    print(f"wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
