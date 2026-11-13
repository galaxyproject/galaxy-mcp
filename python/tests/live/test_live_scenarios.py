"""The live scenario suite: both MCP servers, over stdio, against a real Galaxy.

How the pieces fit, and how to add a scenario: see tests/live/__init__.py.
"""

from __future__ import annotations

import pytest

from .conftest import _ContractBroken
from .loader import Contract
from .model import Pack, Scenario
from .normalise import FindingError, HarnessError
from .runner import (
    ExpectedFailureError,
    LiveContext,
    compare_runs,
    run_scenario,
    unmet_requirements,
)
from .servers import ServerPool

pytestmark = pytest.mark.live


def _unbroken(case: object) -> None:
    if isinstance(case, _ContractBroken):
        pytest.fail(f"the scenario contract does not load: {case.error}")


def _expected_surface(contract: Contract, server: str) -> set[str]:
    names = set(contract.manifest)
    if server == "typescript":
        names = (names - contract.missing_ts_tools()) | contract.missing_py_tools()
    return names


def test_surface(live_server_name: str, live_ctx: LiveContext, live_servers: ServerPool) -> None:
    """tools/list is the manifest, so a stale build or the wrong server fails here first."""
    _unbroken(live_server_name)
    advertised = set(live_servers.get(live_server_name).list_tools())
    expected = _expected_surface(live_ctx.contract, live_server_name)
    missing, extra = sorted(expected - advertised), sorted(advertised - expected)
    hint = ""
    if live_server_name == "python" and "recommend_biocontainer" in missing:
        hint = " (recommend_biocontainer needs the container-recommend extra: uv sync --all-extras)"
    if missing or extra:
        pytest.fail(
            f"{live_server_name} advertises a different surface than the manifest: "
            f"missing {missing}, extra {extra}{hint}",
            pytrace=False,
        )


def test_scenario(
    live_case: tuple[Pack, Scenario, str], live_ctx: LiveContext, live_servers: ServerPool
) -> None:
    _unbroken(live_case)
    pack, scenario, server = live_case
    key = (scenario.id, server)

    unmet = unmet_requirements(live_ctx, scenario)
    for need in unmet:
        if need.skip:
            live_ctx.skipped[key] = need.message
            pytest.skip(need.message)
    # A waiver only skips when it covers every unmet requirement; one it does not cover
    # still fails the run.
    unwaived = [n for n in unmet if n.waiver not in live_ctx.waivers]
    if unwaived:
        pytest.fail(
            "unmet requirements: "
            + "; ".join(f"{n.message} (--live-waive {n.waiver} to skip)" for n in unwaived)
        )
    if unmet:
        live_ctx.skipped[key] = "; ".join(f"waived {n.waiver}: {n.message}" for n in unmet)
        pytest.skip(live_ctx.skipped[key])

    # The message says which step and why; the harness's own frames add nothing, so these
    # are reported without a traceback (and outside the except, so without the chain).
    failure = None
    try:
        session = (
            live_servers.dedicated(server) if scenario.server_env else live_servers.get(server)
        )
        run_scenario(live_ctx, pack, scenario, session)
    except ExpectedFailureError as exc:
        failure = f"xfail: {exc}"
    except (HarnessError, FindingError) as exc:
        failure = f"{type(exc).__name__}: {exc}"
    except AssertionError as exc:
        # A bare assert outside a test module has no message; it is still a failure.
        failure = str(exc) or f"{type(exc).__name__} with no message: {exc!r}"
    if failure is None:
        return
    # galaxy.api_key is a binding, so a tool that echoes its arguments could print it.
    failure = failure.replace(live_ctx.api_key, "<api_key>")
    if failure.startswith("xfail: "):
        pytest.xfail(failure[len("xfail: ") :])
    pytest.fail(failure, pytrace=False)


def test_parity(live_parity_case: tuple[Pack, Scenario], live_ctx: LiveContext) -> None:
    """The two servers' answers, masked, compare equal (shape by default)."""
    _unbroken(live_parity_case)
    _, scenario = live_parity_case
    records = {}
    for server in ("python", "typescript"):
        key = (scenario.id, server)
        if key in live_ctx.skipped:
            pytest.skip(f"{server}: {live_ctx.skipped[key]}")
        record = live_ctx.results.get(key)
        if record is None:
            pytest.fail(f"{scenario.id} never ran on {server}; nothing to compare")
        if not record.completed and record.stopped_by is None:
            pytest.fail(
                f"{scenario.id} did not complete on {server}; "
                f"see test_scenario[{server}-{scenario.id}]"
            )
        records[server] = record

    problems = [
        p.replace(live_ctx.api_key, "<api_key>")
        for p in compare_runs(records["python"], records["typescript"])
    ]
    notes = [f"{s} stopped early: {r.stopped_by}" for s, r in records.items() if r.stopped_by]
    if problems:
        pytest.fail("the servers disagree:\n  " + "\n  ".join(problems + notes), pytrace=False)
