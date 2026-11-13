"""Load scenario packs and the registry, and check everything that can be checked offline.

Every check here returns a list of problems rather than raising, so the offline test can
report all of them at once and the live run can refuse to start on the same list.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .model import (
    FIXTURE_BINDS,
    CallStep,
    LiveRow,
    LiveSection,
    Pack,
    Scenario,
    VerifyStep,
    WaitStep,
    render_schema,
)

REPO = Path(__file__).resolve().parents[3]
CONTRACT = REPO / "contract"
SCENARIOS = CONTRACT / "scenarios"
DATA = SCENARIOS / "data"
SCHEMA_FILE = SCENARIOS / "scenario.schema.json"
MANIFEST = CONTRACT / "mcp-surface.json"
REGISTRY = CONTRACT / "accepted-divergences.json"

# The packs the manifest is split across. Until all of them exist the partition and
# legacy-port checks cannot be complete, and the offline test says so instead of passing.
EXPECTED_PACKS = ("histories-datasets", "tools-jobs", "workflows", "pages-collections-user-tools")

# The tests of test_real_integration.py, which this suite replaced; each is claimed by a `ports`.
LEGACY_TESTS = (
    "TestRealConnection::test_connect_to_galaxy",
    "TestRealConnection::test_get_server_info",
    "TestRealConnection::test_get_current_user",
    "TestRealToolOperations::test_get_tool_panel",
    "TestRealToolOperations::test_search_tools_by_name",
    "TestRealToolOperations::test_get_tool_details",
    "TestRealHistoryOperations::test_get_histories",
    "TestRealHistoryOperations::test_create_history",
    "TestRealHistoryOperations::test_get_history_details",
    "TestRealHistoryOperations::test_get_history_contents",
    "TestRealDatasetOperations::test_upload_and_download_file",
    "TestRealToolExecution::test_run_simple_tool",
    "TestRealIWCOperations::test_get_iwc_workflows",
    "TestRealIWCOperations::test_search_iwc_workflows_rna",
    "TestRealIWCOperations::test_search_iwc_workflows_assembly",
    "TestRealIWCOperations::test_search_iwc_workflows_no_results",
    "TestRealIWCOperations::test_import_workflow_from_iwc",
    "TestRealIWCOperations::test_import_workflow_invalid_trs_id",
    "TestRealIWCOperations::test_get_iwc_workflow_details",
    "TestRealIWCOperations::test_get_iwc_workflow_details_invalid_id",
    "TestRealIWCOperations::test_recommend_iwc_workflows_rnaseq",
    "TestRealIWCOperations::test_recommend_iwc_workflows_assembly",
    "TestRealIWCOperations::test_recommend_iwc_workflows_no_matches",
    "TestRealIWCOperations::test_recommend_iwc_workflows_with_limit",
)

BUILTIN_BINDINGS: dict[str, Any] = {
    "run": {"tag": None, "server": None, "tmp": None},
    "galaxy": {
        "url": None,
        "api_key": None,
        "version": None,
        "user": {"id": None, "email": None, "username": None},
    },
}

REF = re.compile(r"\$\{([^}]*)\}")
REF_PATH = re.compile(r"^[a-z][a-z0-9_]*(\.[A-Za-z0-9_-]+)*$")
LIVE_PREFIX = "live/"
STATIC_PREFIX = "divergences/"
SERVER_CITABLE_KINDS = ("missing-ts-tool", "missing-py-tool")


class LoadError(Exception):
    pass


@dataclass
class Registry:
    raw: dict[str, Any]
    live: LiveSection
    static_rows: dict[str, dict[str, Any]]

    @property
    def live_rows(self) -> dict[str, list[LiveRow]]:
        rows: dict[str, list[LiveRow]] = {}
        for row in self.live.divergences:
            rows.setdefault(row.key, []).append(row)
        return rows

    @property
    def argument_text_enabled(self) -> bool:
        return self.live.normalization.argumentValidationText.enabled


@dataclass
class Contract:
    manifest: dict[str, dict[str, Any]]
    registry: Registry
    packs: list[Pack]
    pack_files: dict[str, Path] = field(default_factory=dict)

    def scenarios(self) -> Iterator[tuple[Pack, Scenario]]:
        for pack in self.packs:
            for scenario in pack.scenarios:
                yield pack, scenario

    def missing_ts_tools(self) -> set[str]:
        return {
            r["tool"] for r in self.registry.static_rows.values() if r["kind"] == "missing-ts-tool"
        }

    def missing_py_tools(self) -> set[str]:
        return {
            r["tool"] for r in self.registry.static_rows.values() if r["kind"] == "missing-py-tool"
        }


def static_key(row: dict[str, Any]) -> str:
    # The same string the TS parity test's divergenceKey() builds, with its runs of spaces
    # collapsed: a row with no param reads "connect :: :: missing-ts-tool", not "::  ::".
    return _spaces(f"{row['tool']} :: {row.get('param') or ''} :: {row['kind']}")


def _spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def load_manifest() -> dict[str, dict[str, Any]]:
    data = json.loads(MANIFEST.read_text())
    return {tool["name"]: tool for tool in data["tools"]}


def load_registry() -> Registry:
    raw = json.loads(REGISTRY.read_text())
    if "live" not in raw:
        raise LoadError(f"{REGISTRY.name} has no `live` section")
    try:
        live = LiveSection.model_validate(raw["live"])
    except ValidationError as exc:
        raise LoadError(f"{REGISTRY.name} -> live: {exc}") from exc
    static_rows = {static_key(row): row for row in raw.get("divergences", [])}
    return Registry(raw=raw, live=live, static_rows=static_rows)


def pack_paths() -> list[Path]:
    return sorted(p for p in SCENARIOS.glob("*.json") if p != SCHEMA_FILE)


def load_pack(path: Path) -> Pack:
    try:
        return Pack.model_validate(json.loads(path.read_text()))
    except (ValidationError, json.JSONDecodeError) as exc:
        raise LoadError(f"{path.relative_to(REPO)}: {exc}") from exc


@cache
def load_contract() -> Contract:
    packs: list[Pack] = []
    files: dict[str, Path] = {}
    for path in pack_paths():
        pack = load_pack(path)
        packs.append(pack)
        files[pack.pack] = path
    return Contract(
        manifest=load_manifest(), registry=load_registry(), packs=packs, pack_files=files
    )


# --- references --------------------------------------------------------------------------


def refs_in(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield from REF.findall(value)
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from refs_in(k)
            yield from refs_in(v)
    elif isinstance(value, list):
        for v in value:
            yield from refs_in(v)


def _check_ref(ref: str, scope: dict[str, Any]) -> str | None:
    """None when `ref` can resolve in `scope`; otherwise why not.

    A scope value of None means "anything below this is fine" (a saved value, whose shape
    only the live run knows); a dict lists the keys that exist.
    """
    if not REF_PATH.match(ref):
        return f"${{{ref}}} is not a reference (binding.key.0.key)"
    parts = ref.split(".")
    node: Any = scope
    for depth, part in enumerate(parts):
        if node is None:
            return None
        if not isinstance(node, dict) or part not in node:
            where = ".".join(parts[:depth]) or "the bindings"
            known = sorted(node) if isinstance(node, dict) else []
            return f"${{{ref}}}: {part!r} is not in {where} (known: {known})"
        node = node[part]
    return None


def fixture_scope(kind: str) -> dict[str, Any]:
    return dict.fromkeys(FIXTURE_BINDS[kind])


def call_steps(scenario: Scenario) -> Iterator[CallStep]:
    for step in scenario.steps:
        if isinstance(step, CallStep):
            yield step


def step_tool(scenario: Scenario, index: int) -> str | None:
    """The tool a step's citations refer to: its own, or for a verify the call before it."""
    step = scenario.steps[index]
    if isinstance(step, CallStep):
        return step.call
    for earlier in reversed(scenario.steps[:index]):
        if isinstance(earlier, CallStep):
            return earlier.call
    return None


def check_references(scenario: Scenario) -> list[str]:
    problems: list[str] = []
    scope: dict[str, Any] = json.loads(json.dumps(BUILTIN_BINDINGS))

    def check(where: str, value: Any) -> None:
        for ref in refs_in(value):
            why = _check_ref(ref, scope)
            if why:
                problems.append(f"{scenario.id} {where}: {why}")

    for fixture in scenario.fixtures:
        check(f"fixture {fixture.as_}", fixture.spec.model_dump(by_alias=True))
        scope[fixture.as_] = fixture_scope(fixture.kind)
    check("server_env", scenario.server_env)
    for i, step in enumerate(scenario.steps):
        where = f"step {i}"
        if isinstance(step, CallStep):
            check(where, step.args)
            check(where, [a.model_dump(by_alias=True, exclude_unset=True) for a in step.expect])
            if step.until:
                check(
                    where,
                    [a.model_dump(by_alias=True, exclude_unset=True) for a in step.until.expect],
                )
            for name in step.save:
                if name in scope:
                    problems.append(f"{scenario.id} {where}: save name {name!r} is already bound")
                scope[name] = None
        elif isinstance(step, WaitStep):
            check(where, step.wait.model_dump())
        elif isinstance(step, VerifyStep):
            check(where, step.verify.target)
            check(where, [a.model_dump(by_alias=True, exclude_unset=True) for a in step.expect])
    return problems


# --- citations ---------------------------------------------------------------------------


def resolve_citation(citation: str, registry: Registry) -> list[LiveRow] | dict[str, Any]:
    """The live rows a citation names (one per server it is about), or the static row."""
    if citation.startswith(LIVE_PREFIX):
        key = citation[len(LIVE_PREFIX) :]
        rows = registry.live_rows.get(key)
        if not rows:
            raise LoadError(f"{citation!r} is not a row in accepted-divergences.json -> live")
        return rows
    if citation.startswith(STATIC_PREFIX):
        key = _spaces(citation[len(STATIC_PREFIX) :])
        static = registry.static_rows.get(key)
        if static is None:
            raise LoadError(
                f"{citation!r} is not a row in accepted-divergences.json -> divergences"
            )
        return static
    raise LoadError(f"{citation!r} must start with {LIVE_PREFIX!r} or {STATIC_PREFIX!r}")


def check_citations(scenario: Scenario, registry: Registry) -> tuple[list[str], set[str]]:
    """Problems with a scenario's citations, and the live keys it cites."""
    problems: list[str] = []
    cited: set[str] = set()
    tools_called = {s.call for s in call_steps(scenario)}

    for citation in scenario.divergences:
        try:
            found = resolve_citation(citation, registry)
        except LoadError as exc:
            problems.append(f"{scenario.id}: {exc}")
            continue
        if not isinstance(found, list):
            if found["kind"] not in SERVER_CITABLE_KINDS or len(scenario.servers) == 2:
                problems.append(
                    f"{scenario.id}: {citation!r} -- a divergences/ row can only justify a "
                    f"servers restriction, and only a {' or '.join(SERVER_CITABLE_KINDS)} row"
                )
            if found["tool"] not in tools_called:
                problems.append(
                    f"{scenario.id}: cites {citation!r} but never calls {found['tool']}"
                )
            continue
        row = found[0]
        cited.add(row.key)
        if row.tool not in tools_called:
            problems.append(f"{scenario.id}: cites {citation!r} but never calls {row.tool}")
        if any(r.expects_failure for r in found):
            problems.append(
                f"{scenario.id}: {citation!r} expects a failure; cite it on the step that fails"
            )

    for i, step in enumerate(scenario.steps):
        if isinstance(step, WaitStep):
            continue
        tool = step_tool(scenario, i)
        for citation in step.divergences:
            try:
                found = resolve_citation(citation, registry)
            except LoadError as exc:
                problems.append(f"{scenario.id} step {i}: {exc}")
                continue
            if not isinstance(found, list):
                problems.append(
                    f"{scenario.id} step {i}: {citation!r} -- only live/ rows apply to a step"
                )
                continue
            row = found[0]
            cited.add(row.key)
            if row.tool != tool:
                problems.append(
                    f"{scenario.id} step {i}: {citation!r} is about {row.tool}, the step's tool "
                    f"is {tool}"
                )
            if not any(r.server in scenario.servers for r in found):
                problems.append(
                    f"{scenario.id} step {i}: {citation!r} rewrites "
                    f"{sorted(r.server for r in found)}, which this scenario never runs"
                )
            if row.where == "file" and not isinstance(step, VerifyStep):
                problems.append(
                    f"{scenario.id} step {i}: {citation!r} is about the file; cite it on a "
                    "verify: file step"
                )
    return problems, cited


def check_servers(scenario: Scenario, registry: Registry) -> list[str]:
    if set(scenario.servers) == {"python", "typescript"}:
        return []
    justified = False
    for citation in scenario.divergences:
        if not citation.startswith(STATIC_PREFIX):
            continue
        row = registry.static_rows.get(_spaces(citation[len(STATIC_PREFIX) :]))
        if row is None or row["kind"] not in SERVER_CITABLE_KINDS:
            continue
        missing_on = "typescript" if row["kind"] == "missing-ts-tool" else "python"
        if missing_on not in scenario.servers:
            justified = True
    if not justified:
        return [
            f"{scenario.id}: runs on {scenario.servers} only; cite the divergences/ row "
            "(missing-ts-tool / missing-py-tool) that explains why"
        ]
    return []


# --- calls against the manifest ----------------------------------------------------------


def _expects_argument_failure(step: CallStep) -> bool:
    return step.outcome == "failure" and any(
        a.path == "failure.kind" and a.op == "equals" and a.equals == "arguments"
        for a in step.expect
    )


def check_calls(scenario: Scenario, manifest: dict[str, dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    for i, step in enumerate(scenario.steps):
        if not isinstance(step, CallStep):
            continue
        where = f"{scenario.id} step {i}"
        tool = manifest.get(step.call)
        if tool is None:
            problems.append(f"{where}: {step.call!r} is not a tool in mcp-surface.json")
            continue
        schema = tool["inputSchema"]
        params = set(schema.get("properties", {}))
        required = set(schema.get("required", []))
        bad_args = _expects_argument_failure(step)
        unknown = set(step.args) - params
        if unknown and not bad_args:
            problems.append(
                f"{where}: {step.call} takes no {sorted(unknown)} (has {sorted(params)})"
            )
        missing = required - set(step.args)
        if missing and not bad_args:
            problems.append(f"{where}: {step.call} needs {sorted(missing)}")
        for kind, path in step.creates.items():
            if not path:
                problems.append(f"{where}: creates.{kind} needs a path in the result")
    return problems


def check_data_files(pack: Pack, scenario: Scenario) -> list[str]:
    problems = []
    for fixture in scenario.fixtures:
        if fixture.workflow is not None:
            path = DATA / pack.pack / fixture.workflow.file
            if not path.is_file():
                problems.append(
                    f"{scenario.id}: fixture {fixture.as_}: no {path.relative_to(REPO)}"
                )
    return problems


# --- whole-contract checks ---------------------------------------------------------------


def check_pack_files(contract: Contract) -> list[str]:
    problems = []
    for name, path in contract.pack_files.items():
        if path.stem != name:
            problems.append(f"{path.name}: `pack` is {name!r}, must match the file name")
    return problems


def check_ids(contract: Contract) -> list[str]:
    seen: dict[str, str] = {}
    problems = []
    for pack, scenario in contract.scenarios():
        if scenario.id in seen:
            problems.append(f"{scenario.id}: in both {seen[scenario.id]} and {pack.pack}")
        seen[scenario.id] = pack.pack
    return problems


def check_ownership(contract: Contract) -> list[str]:
    """Owned tools are disjoint, real, and each called or excused -- never both."""
    problems = []
    owner: dict[str, str] = {}
    for pack in contract.packs:
        called = {s.call for sc in pack.scenarios for s in call_steps(sc)}
        excused = {u.tool for u in pack.uncovered}
        for tool in pack.tools:
            if tool not in contract.manifest:
                problems.append(f"{pack.pack}: owns {tool!r}, which is not in mcp-surface.json")
            if tool in owner:
                problems.append(f"{tool}: owned by both {owner[tool]} and {pack.pack}")
            owner[tool] = pack.pack
            if tool not in called and tool not in excused:
                problems.append(f"{pack.pack}: owns {tool} but no scenario calls it")
        for tool in excused:
            if tool not in pack.tools:
                problems.append(f"{pack.pack}: excuses {tool}, which it does not own")
            if tool in called:
                problems.append(f"{pack.pack}: excuses {tool} but a scenario calls it")
    return problems


def missing_packs(contract: Contract) -> list[str]:
    return [p for p in EXPECTED_PACKS if p not in contract.pack_files]


def check_partition(contract: Contract) -> list[str]:
    owned = {t for p in contract.packs for t in p.tools}
    unowned = sorted(set(contract.manifest) - owned)
    return [f"{tool}: no pack owns it" for tool in unowned]


def check_ports(contract: Contract) -> list[str]:
    claimed: dict[str, str] = {}
    problems = []
    for _, scenario in contract.scenarios():
        for port in scenario.ports:
            if port not in LEGACY_TESTS:
                problems.append(f"{scenario.id}: ports {port!r}, not a test_real_integration test")
            claimed.setdefault(port, scenario.id)
    problems += [f"{name}: no scenario ports it" for name in LEGACY_TESTS if name not in claimed]
    return problems


def check_scenarios(
    contract: Contract, packs: list[Pack] | None = None
) -> tuple[list[str], set[str]]:
    """Problems in the scenarios of `packs` (default all), and the live keys they cite."""
    problems: list[str] = []
    cited: set[str] = set()
    chosen = contract.packs if packs is None else packs
    for pack, scenario in ((p, s) for p in chosen for s in p.scenarios):
        problems += check_references(scenario)
        cite_problems, keys = check_citations(scenario, contract.registry)
        problems += cite_problems
        cited |= keys
        problems += check_servers(scenario, contract.registry)
        problems += check_calls(scenario, contract.manifest)
        problems += check_data_files(pack, scenario)
    return problems, cited


def check_live_registry(contract: Contract, cited: set[str]) -> list[str]:
    live = contract.registry.live
    problems = []
    keys = [(row.key, row.server) for row in live.divergences]
    problems += [
        f"live row {k!r} appears twice for {server}"
        for k, server in {pair for pair in keys if keys.count(pair) > 1}
    ]
    for row in live.divergences:
        if row.tool not in contract.manifest:
            problems.append(f"live row {row.key!r}: {row.tool} is not in mcp-surface.json")
        if row.key not in cited:
            problems.append(f"live row {row.key!r} is cited by no scenario; delete it or cite it")
    want = live.unreviewed_count()
    if live.ratchet.unreviewedGaps != want:
        problems.append(
            f"live.ratchet.unreviewedGaps is {live.ratchet.unreviewedGaps}, the section holds "
            f"{want} unreviewed-gap rows and rules -- set it to {want}"
        )
    return problems


def check_schema_current() -> list[str]:
    if not SCHEMA_FILE.exists() or SCHEMA_FILE.read_text() != render_schema():
        return [
            f"{SCHEMA_FILE.relative_to(REPO)} is out of date: run "
            "`uv run python -m tests.live.model` in python/"
        ]
    return []
