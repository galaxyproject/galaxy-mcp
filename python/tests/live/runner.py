"""Run one scenario against one server, and compare two servers' runs.

`run_scenario` creates the fixtures, walks the steps (call / wait / verify), applies the
registry rows each step cites, and always cleans up. It returns a RunRecord the parity
test reads later; it raises AssertionError on a failed check and ExpectedFailureError when a
cited server-bug row's step failed the way the row says it does.
"""

from __future__ import annotations

import json
import re
import secrets
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import requests
from bioblend.galaxy import GalaxyInstance
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from .assertions import MISSING, evaluate_all, get_path
from .fixtures import FixtureContext, Ledger, make_fixture, verify_view, wait_for
from .loader import REF, Contract, LoadError, resolve_citation, step_tool
from .model import Assertion, CallStep, LiveRow, Pack, Scenario, VerifyStep, WaitStep
from .normalise import (
    GALAXY_ID,
    HarnessError,
    apply_ops,
    apply_step_masks,
    compare_views,
    mask_value,
    to_view,
)
from .servers import StdioSession

NETWORK_PROBES = ("https://raw.githubusercontent.com/", "https://quay.io/")
SESSION_HEX = secrets.token_hex(4)


class ExpectedFailureError(Exception):
    """A step failed exactly where a cited server-bug row says it does."""


@dataclass
class StepRecord:
    index: int
    tool: str
    view: dict[str, Any]
    parity: str
    masks: list[tuple[str, str]]
    # The `where` of an expect-failure row this step failed at: parity skips that path
    # (and below it) and still compares everything else on the step.
    excused: list[str] = field(default_factory=list)
    # The outcome that row documents, when it names one: any other outcome is not excused.
    excused_outcome: str | None = None


@dataclass
class RunRecord:
    scenario_id: str
    server: str
    tag: str
    tmp: str
    galaxy_url: str
    steps: list[StepRecord] = field(default_factory=list)
    bound: list[tuple[str, str]] = field(default_factory=list)
    completed: bool = False
    stopped_by: str | None = None


@dataclass
class LiveContext:
    """What every run shares: the target Galaxy, the contract, and cached probes."""

    gi: GalaxyInstance
    galaxy_url: str
    api_key: str
    version: str
    user: dict[str, Any]
    contract: Contract
    waivers: set[str]
    log_dir: Path
    results: dict[tuple[str, str], RunRecord] = field(default_factory=dict)
    skipped: dict[tuple[str, str], str] = field(default_factory=dict)
    _tools: dict[str, bool] = field(default_factory=dict)

    @cached_property
    def capabilities(self) -> dict[str, bool]:
        return {
            "network": _network_ok(),
            "unprivileged-tools": self._can_create_user_tools(),
            # Listing (and deleting) needs only the user_tool_execute role, not
            # enable_beta_tool_formats.
            "user-tool-role": self._status("unprivileged_tools") == 200,
            "admin": bool(self.user.get("is_admin")),
        }

    def _status(self, route: str) -> int:
        try:
            return int(self.gi.make_get_request(f"{self.gi.url}/{route}").status_code)
        except requests.RequestException:
            return 0

    def _can_create_user_tools(self) -> bool:
        # Listing only needs the user_tool_execute role; creating also needs
        # enable_beta_tool_formats, which Galaxy checks first (403004) and does not expose
        # in /api/configuration. So try a create, and take it straight back.
        if self._status("unprivileged_tools") != 200:
            return False
        probe = {
            "class": "GalaxyUserTool",
            "id": f"mcp_live_probe_{SESSION_HEX}",
            "version": "0.1.0",
            "name": "mcp live probe",
            "container": "busybox",
            "shell_command": "true",
        }
        try:
            reply = self.gi.make_post_request(
                f"{self.gi.url}/unprivileged_tools",
                payload={"src": "representation", "representation": probe},
            )
        except Exception as exc:
            # A lint refusal comes after both permission checks, so it still means yes; a
            # schema refusal (also a 400) means the probe itself is stale, and reads as no
            # so the refusal scenarios run and fail loudly instead.
            return getattr(exc, "status_code", None) == 400 and "lint" in str(exc)
        uuid = reply.get("uuid") if isinstance(reply, dict) else None
        if uuid:
            self.gi.make_delete_request(f"{self.gi.url}/unprivileged_tools/{uuid}")
        return True

    def has_tool(self, tool_id: str) -> bool:
        if tool_id not in self._tools:
            self._tools[tool_id] = self._status(f"tools/{tool_id}") == 200
        return self._tools[tool_id]


def _network_ok() -> bool:
    for url in NETWORK_PROBES:
        try:
            requests.head(url, timeout=10, allow_redirects=True)
        except requests.RequestException:
            return False
    return True


# --- requirements ------------------------------------------------------------------------


@dataclass
class Unmet:
    waiver: str
    message: str
    skip: bool = False


def unmet_requirements(ctx: LiveContext, scenario: Scenario) -> list[Unmet]:
    """What this Galaxy lacks for `scenario`; `skip` marks the negated-capability case."""
    out: list[Unmet] = []
    for cap in scenario.requires.capabilities:
        negated = cap.startswith("!")
        name = cap.lstrip("!")
        have = ctx.capabilities[name]
        if negated and have:
            out.append(
                Unmet(name, f"this Galaxy has {name}; the scenario covers its absence", skip=True)
            )
        elif not negated and not have:
            out.append(Unmet(name, f"this Galaxy lacks capability {name}"))
    for tool_id in scenario.requires.tools:
        if not ctx.has_tool(tool_id):
            out.append(Unmet("tools", f"Galaxy tool {tool_id} is not installed"))
    called = {s.call for s in scenario.steps if isinstance(s, CallStep)}
    for tool in sorted(called):
        spec = (ctx.contract.manifest.get(tool, {}).get("requires") or {}).get("galaxy")
        if spec and not _version_ok(ctx.version, spec):
            out.append(
                Unmet("galaxy-version", f"{tool} needs Galaxy {spec}, this is {ctx.version}")
            )
    return out


def _version_ok(version: str, spec: str) -> bool:
    try:
        return SpecifierSet(spec).contains(Version(version), prereleases=True)
    except InvalidVersion:
        return True


# --- references --------------------------------------------------------------------------


def lookup(bindings: dict[str, Any], ref: str) -> Any:
    node: Any = bindings
    for part in ref.split("."):
        found = get_path(node, part)
        if found is MISSING:
            raise HarnessError(f"${{{ref}}} does not resolve ({part!r} missing)")
        node = found
    return node


def resolve(value: Any, bindings: dict[str, Any]) -> Any:
    if isinstance(value, str):
        whole = REF.fullmatch(value)
        if whole:
            return lookup(bindings, whole.group(1))

        def text(m: re.Match[str]) -> str:
            found = lookup(bindings, m.group(1))
            return found if isinstance(found, str) else json.dumps(found)

        return REF.sub(text, value)
    if isinstance(value, list):
        return [resolve(v, bindings) for v in value]
    if isinstance(value, dict):
        return {resolve(k, bindings): resolve(v, bindings) for k, v in value.items()}
    return value


def resolve_assertions(assertions: list[Assertion], bindings: dict[str, Any]) -> list[Assertion]:
    return [
        Assertion.model_validate(resolve(a.model_dump(by_alias=True, exclude_unset=True), bindings))
        for a in assertions
    ]


def _string_leaves(value: Any, token: str) -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(value, f"<{token}>")] if len(value) >= 8 else []
    if isinstance(value, dict):
        return [x for k, v in value.items() for x in _string_leaves(v, f"{token}.{k}")]
    if isinstance(value, list):
        return [x for i, v in enumerate(value) for x in _string_leaves(v, f"{token}.{i}")]
    return []


# --- one run -----------------------------------------------------------------------------


def run_tag(server: str, scenario_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", scenario_id)[:48].strip("-")
    return f"mcp-live-{SESSION_HEX}-{server}-{slug}"


class _Rows:
    """The registry rows a scenario cites, and which of them ever changed anything."""

    def __init__(self, ctx: LiveContext, scenario: Scenario, server: str) -> None:
        self.ctx = ctx
        self.server = server
        self.scenario_rows = [
            r for c in scenario.divergences if c.startswith("live/") for r in self._rows(c)
        ]
        self.scenario_changed: set[str] = set()

    def _rows(self, citation: str) -> list[LiveRow]:
        try:
            found = resolve_citation(citation, self.ctx.contract.registry)
        except LoadError as exc:
            raise HarnessError(str(exc)) from exc
        if not isinstance(found, list):
            raise HarnessError(f"{citation!r} is not a live row")
        return found

    def for_step(
        self, citations: list[str], tool: str | None
    ) -> tuple[list[LiveRow], list[LiveRow]]:
        step = [r for c in citations for r in self._rows(c)]
        for row in step:
            if row.tool != tool:
                raise HarnessError(f"live/{row.key} is cited on a {tool} step")
        scen = [r for r in self.scenario_rows if r.tool == tool]
        mine = lambda rows: [r for r in rows if r.server == self.server]  # noqa: E731
        return mine(step), mine(scen)


def _adapt(
    view: Any, step_rows: list[LiveRow], scen_rows: list[LiveRow], rows: _Rows
) -> tuple[Any, list[str], LiveRow | None]:
    """Apply the rewriting rows; stale step rows, and the expect-failure row if any."""
    stale: list[str] = []
    expect_row: LiveRow | None = None
    for row in step_rows + scen_rows:
        if row.expects_failure:
            expect_row = row
            continue
        adapted = apply_ops(view, row)
        changed = adapted != view
        if row in step_rows and not changed:
            stale.append(row.key)
        if changed and row in scen_rows:
            rows.scenario_changed.add(row.key)
        view = adapted
    return view, stale, expect_row


def _stale_message(server: str, keys: list[str]) -> str:
    return (
        f"stale registry rows on {server}: {keys} changed nothing on this step. The servers "
        "may have converged -- check, then update or delete the row."
    )


def run_scenario(
    ctx: LiveContext, pack: Pack, scenario: Scenario, session: StdioSession
) -> RunRecord:
    server = session.server
    tag = run_tag(server, scenario.id)
    tmp = Path(tempfile.mkdtemp(prefix=f"{tag}-")).resolve()
    record = RunRecord(scenario.id, server, tag, str(tmp), ctx.galaxy_url)
    ctx.results[(scenario.id, server)] = record
    ledger = Ledger(ctx.gi)
    deadline = time.monotonic() + scenario.timeout
    bindings: dict[str, Any] = {
        "run": {"tag": tag, "server": server, "tmp": str(tmp)},
        "galaxy": {
            "url": ctx.galaxy_url,
            "api_key": ctx.api_key,
            "version": ctx.version,
            "user": {k: ctx.user.get(k) for k in ("id", "email", "username")},
        },
    }
    record.bound += [
        (tag, "<tag>"),
        (str(tmp), "<tmp>"),
        (ctx.galaxy_url, "<galaxy>"),
        (ctx.api_key, "<api_key>"),
    ]
    rows = _Rows(ctx, scenario, server)
    fctx = FixtureContext(ctx.gi, ledger, tag, tmp, pack.pack, timeout=180)
    primary: BaseException | None = None
    try:
        if not scenario.server_env:
            session.ensure_alive()
        for fixture in scenario.fixtures:
            spec = type(fixture.spec).model_validate(
                resolve(fixture.spec.model_dump(by_alias=True, exclude_unset=True), bindings)
            )
            bound = make_fixture(fctx, fixture, spec)
            bindings[fixture.as_] = bound
            record.bound += _string_leaves(bound, fixture.as_)

        if scenario.server_env:
            # A server of this scenario's own (see ServerPool.dedicated), started only now
            # because its environment names things the fixtures just made.
            session.env.update(resolve(scenario.server_env, bindings))
            session.ensure_alive()

        for index, step in enumerate(scenario.steps):
            if time.monotonic() > deadline:
                raise AssertionError(
                    f"scenario timeout ({scenario.timeout:.0f}s) before step {index}"
                )
            where = f"[{server}] step {index}"
            if isinstance(step, WaitStep):
                ident = resolve(getattr(step.wait, step.wait.kind), bindings)
                wait_for(ctx.gi, step.wait, ident, deadline)
            elif isinstance(step, VerifyStep):
                _verify(ctx, scenario, index, step, bindings, rows, where)
            else:
                _call(
                    ctx,
                    scenario,
                    index,
                    step,
                    bindings,
                    rows,
                    where,
                    session,
                    ledger,
                    record,
                    deadline,
                )

        never = [
            r.key
            for r in rows.scenario_rows
            if r.server == server and not r.expects_failure and r.key not in rows.scenario_changed
        ]
        if never:
            raise AssertionError(_stale_message(server, never) + " (cited by the scenario)")
        record.completed = True
        return record
    except ExpectedFailureError as exc:
        record.stopped_by = str(exc)
        primary = exc
        raise
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if scenario.server_env:
            session.close()
        errors = ledger.cleanup()
        shutil.rmtree(tmp, ignore_errors=True)
        if errors:
            message = "cleanup left things on the shared Galaxy:\n  " + "\n  ".join(errors)
            # A known server bug excuses the step, never a leak: an xfail turns into a
            # failure here. A real failure is already failing the test, so it keeps its own
            # message and the leak is printed beside it.
            if primary is None or isinstance(primary, ExpectedFailureError):
                if primary is not None:
                    record.stopped_by = None
                    message += f"\n(after an expected failure: {primary})"
                raise AssertionError(message) from None
            print(message)


def _verify(
    ctx: LiveContext,
    scenario: Scenario,
    index: int,
    step: VerifyStep,
    bindings: dict[str, Any],
    rows: _Rows,
    where: str,
) -> None:
    target = resolve(step.verify.target, bindings)
    view = verify_view(ctx.gi, step.verify.kind, target)
    step_rows, scen_rows = rows.for_step(step.divergences, step_tool(scenario, index))
    view, stale, expect_row = _adapt(view, step_rows, scen_rows, rows)
    failures = evaluate_all(view, resolve_assertions(step.expect, bindings))
    _conclude(
        where,
        f"verify {step.verify.kind} {target}",
        view,
        failures,
        stale,
        expect_row,
        rows.server,
        whole_file=step.verify.kind == "file",
    )


def _bare_path(path: str) -> str:
    """A view path without list indexes or a `where` bracket qualifier."""
    path = re.sub(r"\[[^\]]*\]$", "", path)
    return ".".join(seg for seg in path.split(".") if not seg.isdigit())


def unexplained_failures(failures: list[str], row: LiveRow, whole_file: bool) -> list[str]:
    """The failures an expect-failure row does not account for.

    A row explains failures at its own `where` (or below it); anything else on the step --
    an outcome flip, a missing `creates`, another assertion -- is a new problem that must not
    hide behind the known one. `where: file` covers a whole verify:file view.
    """
    if row.where == "file" and whole_file:
        return []
    where = _bare_path(row.where)
    out = []
    for failure in failures:
        path = _bare_path(failure.split(": ", 1)[0])
        if not (path == where or path.startswith(where + ".")):
            out.append(failure)
    return out


def _conclude(
    where: str,
    what: str,
    view: Any,
    failures: list[str],
    stale: list[str],
    expect_row: LiveRow | None,
    server: str,
    whole_file: bool = False,
) -> None:
    if expect_row is not None:
        documented = expect_row.documented_outcome
        if documented and outcome_of(view) != documented:
            raise AssertionError(
                f"{where} {what} cites live/{expect_row.key}, which documents {documented} on "
                f"{server}, but the step answered {outcome_of(view)}: {_failure_hint(view)}"
            )
        other = unexplained_failures(failures, expect_row, whole_file)
        if other:
            raise AssertionError(
                f"{where} {what} cites live/{expect_row.key}, which expects a failure at "
                f"{expect_row.where}, but it also failed elsewhere:\n  " + "\n  ".join(other)
            )
        if failures:
            raise ExpectedFailureError(
                f"{where} {what} failed as live/{expect_row.key} expects ({expect_row.status}): "
                f"{expect_row.reason} | first failure: {failures[0]}"
            )
        raise AssertionError(
            f"{where} {what} passed, but live/{expect_row.key} says it fails on {server}. "
            "Fixed? Delete the row and its citation."
        )
    if failures:
        shown = json.dumps(view, indent=1, default=str)
        if len(shown) > 4000:
            shown = shown[:4000] + "\n..."
        raise AssertionError(f"{where} {what}:\n  " + "\n  ".join(failures) + f"\nview:\n{shown}")
    if stale:
        raise AssertionError(f"{where} {what}: " + _stale_message(server, stale))


def outcome_of(view: Any) -> str:
    """ "success", "failure:<kind>", or "-" for a verify view."""
    if not isinstance(view, dict) or "outcome" not in view:
        return "-"
    if view["outcome"] == "failure":
        return f"failure:{view['failure']['kind']}"
    return str(view["outcome"])


def _failure_hint(view: Any) -> str:
    if isinstance(view, dict) and view.get("outcome") == "failure":
        return str(view["failure"]["sentence"])[:300]
    return str(view.get("message") if isinstance(view, dict) else view)[:300]


def _outcome_failures(step: CallStep, view: dict[str, Any]) -> list[str]:
    if view["outcome"] == step.outcome:
        return []
    detail = view["failure"]["sentence"] if view["outcome"] == "failure" else view.get("message")
    return [f"outcome: expected {step.outcome}, got {view['outcome']}: {str(detail)[:1500]}"]


def _call(
    ctx: LiveContext,
    scenario: Scenario,
    index: int,
    step: CallStep,
    bindings: dict[str, Any],
    rows: _Rows,
    where: str,
    session: StdioSession,
    ledger: Ledger,
    record: RunRecord,
    deadline: float,
) -> None:
    args = resolve(step.args, bindings)
    step_rows, scen_rows = rows.for_step(step.divergences, step.call)
    argument_text = ctx.contract.registry.argument_text_enabled

    def once() -> tuple[dict[str, Any], dict[str, Any], list[str], LiveRow | None]:
        remaining = max(5.0, deadline - time.monotonic())
        result = session.call(step.call, args, timeout=min(300.0, remaining))
        raw = to_view(step.call, result, session.server, argument_text)
        if raw["outcome"] == "success":
            for kind, path in step.creates.items():
                found = get_path(raw, path)
                if isinstance(found, str):
                    ledger.add(kind, found)
        adapted, stale, expect_row = _adapt(raw, step_rows, scen_rows, rows)
        return raw, adapted, stale, expect_row

    raw, view, stale, expect_row = once()
    if step.until is not None:
        until = resolve_assertions(step.until.expect, bindings)
        stop = min(time.monotonic() + step.until.timeout, deadline)
        while True:
            pending = _outcome_failures(step, view) or evaluate_all(view, until)
            if not pending:
                break
            if time.monotonic() > stop:
                raise AssertionError(
                    f"{where} {step.call}: `until` still unmet after {step.until.timeout:.0f}s: "
                    + "; ".join(pending)
                )
            time.sleep(step.until.interval)
            raw, view, stale, expect_row = once()

    if raw["outcome"] == "success":
        missing = [
            f"creates.{k}: nothing at {p}"
            for k, p in step.creates.items()
            if not isinstance(get_path(raw, p), str)
        ]
    else:
        missing = []
    failures = missing + _outcome_failures(step, view)
    if not failures:
        failures = evaluate_all(view, resolve_assertions(step.expect, bindings))

    def keep(excused: list[str], documented: str | None = None) -> None:
        masks = [(m.path, resolve(m.pattern, bindings)) for m in step.mask]
        record.steps.append(
            StepRecord(index, step.call, view, step.parity, masks, excused, documented)
        )

    try:
        _conclude(where, step.call, view, failures, stale, expect_row, session.server)
    except ExpectedFailureError:
        # Kept for parity, so the rest of the step's answer is still compared.
        assert expect_row is not None
        keep([expect_row.where], expect_row.documented_outcome)
        raise

    for name, path in step.save.items():
        value = get_path(view, path)
        if value is MISSING:
            raise AssertionError(f"{where} {step.call}: save {name}: nothing at {path}")
        bindings[name] = value
        record.bound += _string_leaves(value, name)
    keep([])


# --- parity ------------------------------------------------------------------------------


def _masked(record: RunRecord, view: dict[str, Any]) -> Any:
    # The first name a value was bound under wins, so a value bound twice (a fixture id a
    # step also saves) masks to the same token on both servers.
    first: dict[str, str] = {}
    for value, token in record.bound:
        first.setdefault(value, token)
    replacements = sorted(first.items(), key=lambda pair: (-len(pair[0]), pair[1]))
    return mask_value(view, replacements)


def _id_tokens(*records: RunRecord) -> frozenset[str]:
    return frozenset(
        token for r in records for value, token in r.bound if GALAXY_ID.fullmatch(value)
    )


def _same_value(record: RunRecord) -> dict[str, set[str]]:
    """Each token, and every token this run bound to one of its values (itself included)."""
    values: dict[str, set[str]] = {}
    tokens: dict[str, set[str]] = {}
    for value, token in record.bound:
        values.setdefault(token, set()).add(value)
        tokens.setdefault(value, set()).add(token)
    return {t: {u for v in vs for u in tokens[v]} for t, vs in values.items()}


def _aliases(py: RunRecord, ts: RunRecord) -> frozenset[tuple[str, str]]:
    """(python token, typescript token) pairs that may name the same binding.

    _masked shows a value under the first token bound to it, so when an HDA happens to share
    an encoded id with a history, a correct answer can read <h.id> on one server and <gamma>
    or <c.id> on the other. The two tokens match when some name N links them: python bound
    N to the value its token stands for, and typescript bound N to the value its token
    stands for. Tokens with no such N -- different values on both runs -- still differ.
    """
    py_same, ts_same = _same_value(py), _same_value(ts)
    out: set[tuple[str, str]] = set()
    for name in py_same.keys() & ts_same.keys():
        out.update((x, y) for x in py_same[name] for y in ts_same[name])
    return frozenset(out)


def compare_runs(py: RunRecord, ts: RunRecord) -> list[str]:
    problems: list[str] = []
    weak = _id_tokens(py, ts)
    aliases = _aliases(py, ts)
    by_index = {s.index: s for s in ts.steps}
    for p in py.steps:
        t = by_index.get(p.index)
        if t is None or p.parity == "none":
            continue
        pv, tv = _masked(py, p.view), _masked(ts, t.view)
        if p.masks:
            pv, phits = apply_step_masks(pv, p.masks)
            tv, thits = apply_step_masks(tv, t.masks)
            for (path, pattern), ph, th in zip(p.masks, phits, thits, strict=True):
                if not ph or not th:
                    problems.append(
                        f"step {p.index} {p.tool}: mask {pattern!r} at {path} matched on "
                        f"python {ph}x, typescript {th}x -- a mask must match on both servers"
                    )
        for step, server in ((p, "python"), (t, "typescript")):
            if step.excused_outcome and outcome_of(step.view) != step.excused_outcome:
                problems.append(
                    f"step {p.index} {p.tool}: {server} answered {outcome_of(step.view)}, but "
                    f"its expect-failure row documents {step.excused_outcome}"
                )
        excused = {_bare_path(w) for w in p.excused + t.excused}
        for diff in compare_views(pv, tv, p.parity, weak, aliases):
            if not _diff_excused(diff, excused):
                problems.append(f"step {p.index} {p.tool} [{p.parity}] {diff}")
    return problems


def _diff_excused(diff: str, excused: set[str]) -> bool:
    path = diff.split(":", 1)[0].replace("[]", "")
    path = _bare_path(path)
    return any(path == w or path.startswith(w + ".") for w in excused)
