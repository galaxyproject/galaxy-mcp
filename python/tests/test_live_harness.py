"""Offline tests for the live harness's own machinery: the normaliser, registry adapters,
staleness and expected failures, parity masking and shape comparison, references.

The live run can only show these working on whatever the servers happen to answer today;
these pin down what each rule does with the answers it is there to catch.
"""

import json
import os
import time

import pytest
from mcp.types import CallToolResult, TextContent

from tests.live import servers
from tests.live.assertions import evaluate_all
from tests.live.model import Assertion, LiveRow
from tests.live.normalise import (
    FindingError,
    HarnessError,
    apply_ops,
    compare_views,
    mask_value,
    to_view,
)
from tests.live.runner import (
    ExpectedFailureError,
    RunRecord,
    StepRecord,
    _conclude,
    compare_runs,
    resolve,
    unexplained_failures,
)


def _result(text, *, error=False, structured=None):
    return CallToolResult(
        content=[TextContent(type="text", text=text)], isError=error, structuredContent=structured
    )


def _envelope(**over):
    env = {"data": {"x": 1}, "success": True, "message": "ok", "count": None, "pagination": None}
    env.update(over)
    return env


def _row(**over):
    row = {
        "tool": "get_server_info",
        "where": "data.url",
        "kind": "value-mismatch",
        "server": "python",
        "observed": "o",
        "adapter": {"rstrip": {"paths": ["data.url"], "chars": "/"}},
        "status": "unreviewed-gap",
        "reason": "r",
    }
    row.update(over)
    return LiveRow.model_validate(row)


class TestToView:
    def test_tool_failure_strips_the_prefix(self):
        r = _result("Error calling tool 'get_user': Galaxy said no", error=True)
        assert to_view("get_user", r, "python", True) == {
            "outcome": "failure",
            "failure": {"kind": "tool", "sentence": "Galaxy said no"},
        }

    @pytest.mark.parametrize(
        "text",
        [
            "1 validation error for call[get_histories]\nlimit\n  bad",
            "2 validation errors for call[get_histories]\n...",
            "MCP error -32602: Input validation error: Invalid arguments for tool get_histories",
        ],
    )
    def test_argument_failures_from_either_framework(self, text):
        view = to_view("get_histories", _result(text, error=True), "x", True)
        assert view["failure"]["kind"] == "arguments"

    def test_argument_text_needs_the_registry_rule(self):
        r = _result("1 validation error for call[get_histories]\n", error=True)
        with pytest.raises(HarnessError, match="disabled"):
            to_view("get_histories", r, "python", False)

    def test_pydantic_text_for_another_tool_is_not_an_argument_failure(self):
        r = _result("1 validation error for call[other_tool]\n", error=True)
        with pytest.raises(HarnessError, match="unrecognised"):
            to_view("get_histories", r, "python", True)

    def test_unrecognised_error_text(self):
        with pytest.raises(HarnessError, match="unrecognised"):
            to_view("get_user", _result("boom", error=True), "typescript", True)

    def test_success_reads_the_text_block(self):
        env = _envelope()
        view = to_view("get_user", _result(json.dumps(env)), "typescript", True)
        assert view == {"outcome": "success", **env}

    def test_structured_content_must_match_the_text(self):
        env = _envelope()
        r = _result(json.dumps(env), structured=_envelope(message="different"))
        with pytest.raises(FindingError, match="disagree"):
            to_view("get_user", r, "python", True)

    def test_structured_content_compares_numbers_by_value(self):
        r = _result(json.dumps(_envelope(data={"n": 0})), structured=_envelope(data={"n": 0.0}))
        assert to_view("get_user", r, "python", True)["data"] == {"n": 0}

    def test_success_false_in_a_success_result(self):
        with pytest.raises(FindingError, match="success=False"):
            to_view("get_user", _result(json.dumps(_envelope(success=False))), "python", True)

    def test_envelope_keys_are_exact(self):
        env = {**_envelope(), "extra": 1}
        with pytest.raises(FindingError, match="envelope keys"):
            to_view("get_user", _result(json.dumps(env)), "python", True)

    def test_more_than_one_content_block(self):
        r = CallToolResult(
            content=[TextContent(type="text", text="{}"), TextContent(type="text", text="{}")]
        )
        with pytest.raises(FindingError, match="exactly one"):
            to_view("get_user", r, "python", True)


class TestAdapters:
    def test_rstrip(self):
        view = {"data": {"url": "http://g/"}}
        assert apply_ops(view, _row())["data"]["url"] == "http://g"
        assert view["data"]["url"] == "http://g/", "adapters must not mutate their input"

    def test_drop_with_wildcard(self):
        row = _row(adapter={"drop": ["data.items.*.note", "message"]})
        view = {"data": {"items": [{"a": 1, "note": "x"}, {"a": 2, "note": "y"}]}, "message": "m"}
        assert apply_ops(view, row) == {"data": {"items": [{"a": 1}, {"a": 2}]}}

    def test_sub_and_cut_from(self):
        row = _row(
            adapter=[
                {"sub": [{"path": "message", "pattern": r"file '.*/([^/]+)'", "replace": r"\1"}]},
                {"cut_from": {"path": "failure.sentence", "marker": "\n\nWorkflow input slots:"}},
            ]
        )
        view = {
            "message": "Uploaded file '/tmp/x/a.txt'",
            "failure": {"sentence": "refused\n\nWorkflow input slots: {}"},
        }
        assert apply_ops(view, row) == {
            "message": "Uploaded a.txt",
            "failure": {"sentence": "refused"},
        }

    def test_expect_failure_stands_alone(self):
        with pytest.raises(ValueError, match="on its own"):
            _row(adapter=[{"expect": "failure"}, {"drop": ["x"]}])


class TestConclude:
    def test_stale_row_fails(self):
        with pytest.raises(AssertionError, match="stale registry rows"):
            _conclude(
                "w", "t", {}, [], ["get_server_info :: data.url :: value-mismatch"], None, "python"
            )

    def test_expected_failure_that_fails_is_reported_as_expected(self):
        from tests.live.runner import ExpectedFailureError

        row = _row(kind="server-bug", where="file", adapter={"expect": "failure"})
        with pytest.raises(ExpectedFailureError, match="failed as live/"):
            _conclude("w", "t", {}, ["content: expected 'x'"], [], row, "python", whole_file=True)

    def test_expected_failure_only_explains_its_own_path(self):
        from tests.live.runner import ExpectedFailureError

        row = _row(
            kind="server-bug", where="data.elements.extension", adapter={"expect": "failure"}
        )
        with pytest.raises(ExpectedFailureError):
            _conclude("w", "t", {}, ["data.elements.0.extension: x"], [], row, "python")
        for other in ("outcome: expected success, got failure: boom", "data.collection.state: x"):
            with pytest.raises(AssertionError, match="also failed elsewhere"):
                _conclude("w", "t", {}, ["data.elements.1.extension: x", other], [], row, "python")

    def test_bracket_qualified_where_still_matches_its_path(self):
        row = _row(kind="server-bug", where="failure.sentence[x]", adapter={"expect": "failure"})
        assert unexplained_failures(["failure.sentence: y"], row, whole_file=False) == []

    def test_expected_failure_that_passes_asks_for_the_row_to_go(self):
        row = _row(kind="server-bug", where="file", adapter={"expect": "failure"})
        with pytest.raises(AssertionError, match="Delete the row"):
            _conclude("w", "t", {}, [], [], row, "python")


class TestParity:
    def test_url_slash_survives_masking(self):
        url = "http://localhost:8091"
        repl = [(url, "<galaxy>")]
        assert mask_value({"u": url + "/"}, repl) == {"u": "<galaxy>/"}

    def test_ids_times_uuids_and_keys_are_masked(self):
        view = {
            "id": "0123456789abcdef",
            "t": "2026-10-09T03:25:34.297972",
            "u": "6bf79aec-9330-4517-bf8e-88821ba5b86d",
            "by_id": {"fedcba9876543210": 1},
        }
        assert mask_value(view, []) == {
            "id": "<id>",
            "t": "<time>",
            "u": "<uuid>",
            "by_id": {"<id>": 1},
        }

    def test_shape_treats_numbers_alike_and_empty_lists_as_wildcards(self):
        py = {"outcome": "success", "data": {"n": 0.0, "z": None, "l": [], "m": [None]}}
        ts = {"outcome": "success", "data": {"n": 0, "z": None, "l": [{"a": 1}], "m": ["x"]}}
        assert compare_views(py, ts, "shape") == []

    def test_shape_reports_null_against_a_value(self):
        py = {"outcome": "success", "data": {"suggested_filename": None}}
        ts = {"outcome": "success", "data": {"suggested_filename": "a.txt"}}
        assert compare_views(py, ts, "shape") == [
            "data.suggested_filename: python null, typescript string"
        ]

    def test_shape_compares_the_message(self):
        py = {"outcome": "success", "message": "Uploaded file from URL 'u'"}
        ts = {"outcome": "success", "message": "Submitted URL upload to history <h.id>"}
        (diff,) = compare_views(py, ts, "shape")
        assert diff.startswith("message: python")

    def test_shape_reports_key_differences(self):
        py = {"outcome": "success", "data": {"note": "x", "file_size": 0}}
        ts = {"outcome": "success", "data": {}}
        (diff,) = compare_views(py, ts, "shape")
        assert "only on python ['file_size', 'note']" in diff

    def test_exact_reports_values(self):
        py = {"outcome": "success", "message": "a/"}
        ts = {"outcome": "success", "message": "a"}
        assert compare_views(py, ts, "exact") == ['message: python "a/", typescript "a"']

    def test_failure_sentences_compare_but_argument_text_does_not(self):
        tool = {"outcome": "failure", "failure": {"kind": "tool", "sentence": "a"}}
        other = {"outcome": "failure", "failure": {"kind": "tool", "sentence": "b"}}
        assert compare_views(tool, other, "shape")
        args_py = {"outcome": "failure", "failure": {"kind": "arguments", "sentence": "pydantic"}}
        args_ts = {"outcome": "failure", "failure": {"kind": "arguments", "sentence": "-32602"}}
        assert compare_views(args_py, args_ts, "shape") == []

    def test_outcome_difference(self):
        ok = {"outcome": "success", "message": "m"}
        bad = {"outcome": "failure", "failure": {"kind": "tool", "sentence": "s"}}
        assert compare_views(ok, bad, "shape")[0].startswith("outcome:")


class TestReferences:
    bindings = {"h": {"id": "abc", "n": 3}, "run": {"tag": "t"}}

    def test_whole_reference_keeps_type(self):
        assert resolve("${h.n}", self.bindings) == 3
        assert resolve({"k": ["${h}"]}, self.bindings) == {"k": [{"id": "abc", "n": 3}]}

    def test_interpolation(self):
        assert resolve("${run.tag}-x-${h.n}", self.bindings) == "t-x-3"

    def test_unresolved(self):
        with pytest.raises(HarnessError, match="does not resolve"):
            resolve("${h.nope}", self.bindings)


class TestAssertions:
    view = {
        "data": {"items": [{"id": "a", "score": 3}, {"id": "b", "score": 1}], "n": 2.0},
        "pagination": {"returned": 2, "limit": 5},
    }

    @pytest.mark.parametrize(
        "assertion",
        [
            {"path": "data.n", "equals": 2},
            {"path": "data.n", "type": "integer"},
            {"path": "data.items", "has_item": {"id": "b"}},
            {"path": "data.items", "sorted": {"key": "score", "order": "desc"}},
            {"path": "data.items", "each": [{"path": "id", "type": "string"}]},
            {"path": "pagination.returned", "lte": {"path": "pagination.limit"}},
            {"path": "data.items", "length": {"gte": 2, "lt": 3}},
            {"path": "data.items.1.id", "in": ["a", "b"]},
            {"path": "data.missing", "exists": False},
            {"path": "", "keys_equal": ["data", "pagination"]},
        ],
    )
    def test_holds(self, assertion):
        assert evaluate_all(self.view, [Assertion.model_validate(assertion)]) == []

    @pytest.mark.parametrize(
        "assertion",
        [
            {"path": "data.n", "equals": True},
            {"path": "data.items", "sorted": {"key": "score"}},
            {"path": "data.items", "has_item": {"id": "c"}},
            {"path": "data.missing", "equals": None},
        ],
    )
    def test_fails(self, assertion):
        assert evaluate_all(self.view, [Assertion.model_validate(assertion)])

    def test_exactly_one_op(self):
        with pytest.raises(ValueError, match="exactly one"):
            Assertion.model_validate({"path": "x", "equals": 1, "contains": "y"})


def test_stale_ts_build_is_detected(tmp_path, monkeypatch):
    for name in servers.TS_BUILT_PACKAGES:
        (tmp_path / name / "src").mkdir(parents=True)
        (tmp_path / name / "dist").mkdir(parents=True)
        (tmp_path / name / "src" / "a.ts").write_text("")
        (tmp_path / name / "dist" / "a.js").write_text("")
    entry = tmp_path / "galaxy-mcp" / "dist" / "index.js"
    entry.write_text("")
    monkeypatch.setattr(servers, "TS_PACKAGES", tmp_path)
    monkeypatch.setattr(servers, "TS_ENTRY", entry)
    monkeypatch.setattr(servers, "REPO", tmp_path)
    old = time.time() - 100
    for name in servers.TS_BUILT_PACKAGES:
        os.utime(tmp_path / name / "src" / "a.ts", (old, old))
    assert servers.ts_build_problem() is None
    os.utime(tmp_path / "galaxy-ops" / "src" / "a.ts", None)
    os.utime(tmp_path / "galaxy-ops" / "dist" / "a.js", (old - 10, old - 10))
    os.utime(entry, (old - 10, old - 10))
    assert "galaxy-ops/dist is older" in (servers.ts_build_problem() or "")


class TestIntegrationAdditions:
    def test_set_overwrites_a_boolean_and_is_stale_when_equal(self):
        row = _row(
            tool="download_dataset",
            where="data.content_available",
            server="typescript",
            adapter={"set": [{"path": "data.content_available", "from": False, "to": True}]},
        )
        view = {"outcome": "success", "data": {"content_available": False}}
        assert apply_ops(view, row)["data"]["content_available"] is True
        already = {"outcome": "success", "data": {"content_available": True}}
        assert apply_ops(already, row) == already

    def test_set_only_swaps_the_documented_value(self):
        row = _row(
            tool="download_dataset",
            where="data.suggested_filename",
            server="typescript",
            adapter={"set": [{"path": "data.suggested_filename", "from": "a.txt", "to": None}]},
        )
        view = {"outcome": "success", "data": {"suggested_filename": "a.txt"}}
        assert apply_ops(view, row)["data"]["suggested_filename"] is None
        for wrong in (123, "wrong-name", 0, False):
            other = {"outcome": "success", "data": {"suggested_filename": wrong}}
            assert apply_ops(other, row) == other

    def test_set_never_adds_a_missing_key(self):
        row = _row(adapter={"set": [{"path": "data.absent", "from": None, "to": 1}]})
        view = {"outcome": "success", "data": {}}
        assert apply_ops(view, row) == view

    def test_set_needs_a_from(self):
        with pytest.raises(ValueError, match="from"):
            _row(adapter={"set": [{"path": "data.x", "to": 1}]})

    def test_bound_id_token_matches_a_plain_id_but_not_another_binding(self):
        py = {"outcome": "success", "data": {"a": "<id>", "b": "<d2.id>"}}
        ts = {"outcome": "success", "data": {"a": "<d2.id>", "b": "<d3.id>"}}
        weak = frozenset({"<d2.id>", "<d3.id>"})
        assert compare_views(py, ts, "exact", weak) == [
            'data.b: python "<d2.id>", typescript "<d3.id>"'
        ]

    def test_numeric_bound_from_a_reference(self):
        raw = Assertion.model_validate({"path": "n", "gte": "${count}"})
        resolved = Assertion.model_validate(
            resolve(raw.model_dump(by_alias=True, exclude_unset=True), {"count": 3})
        )
        assert evaluate_all({"n": 3}, [resolved]) == []
        assert evaluate_all({"n": 2}, [resolved])

    def test_a_non_reference_string_bound_is_refused(self):
        with pytest.raises(ValueError):
            Assertion.model_validate({"path": "n", "gte": "three"})

    def test_one_key_holds_a_row_per_server(self):
        from tests.live.loader import Registry, resolve_citation
        from tests.live.model import LiveSection

        rows = [
            _row(kind="server-bug", adapter={"expect": "failure"}),
            _row(kind="server-bug", adapter={"expect": "failure"}, server="typescript"),
        ]
        live = LiveSection.model_validate(
            {
                "ratchet": {"unreviewedGaps": 3},
                "normalization": {
                    "argumentValidationText": {
                        "enabled": True,
                        "status": "unreviewed-gap",
                        "reason": "r",
                    }
                },
                "divergences": [r.model_dump(by_alias=True) for r in rows],
            }
        )
        registry = Registry(raw={}, live=live, static_rows={})
        found = resolve_citation("live/get_server_info :: data.url :: server-bug", registry)
        assert isinstance(found, list)
        assert sorted(r.server for r in found) == ["python", "typescript"]

    def test_static_citation_ignores_runs_of_spaces(self):
        from tests.live.loader import Registry, resolve_citation, static_key

        row = {"tool": "connect", "kind": "missing-ts-tool"}
        registry = Registry(raw={}, live=None, static_rows={static_key(row): row})  # type: ignore[arg-type]
        for spelling in (
            "divergences/connect :: :: missing-ts-tool",
            "divergences/connect ::  :: missing-ts-tool",
        ):
            assert resolve_citation(spelling, registry) == row


def _run(server, steps):
    record = RunRecord("s.x", server, "tag", "/tmp/x", "http://g")
    record.steps = steps
    return record


class TestExpectedFailureParity:
    def _pair(self, ts_data, ts_message="m", excused=("data.citations",)):
        py = {"outcome": "success", "data": {"citations": []}, "message": "m"}
        ts = {"outcome": "success", "data": ts_data, "message": ts_message}
        return (
            _run("python", [StepRecord(0, "t", py, "exact", [], list(excused))]),
            _run("typescript", [StepRecord(0, "t", ts, "exact", [], list(excused))]),
        )

    def test_the_excused_path_is_left_out(self):
        py, ts = self._pair({"citations": [{"x": 1}]})
        assert compare_runs(py, ts) == []

    def test_everything_else_on_the_step_is_still_compared(self):
        py, ts = self._pair({"citations": [], "extra": 1}, ts_message="other")
        problems = compare_runs(py, ts)
        assert any("keys only on" in p and "extra" in p for p in problems)
        assert any("message" in p for p in problems)

    def test_one_side_excused_covers_the_pair(self):
        py, ts = self._pair({"citations": [{"x": 1}]}, excused=())
        ts.steps[0].excused = ["data.citations[qualifier]"]
        assert compare_runs(py, ts) == []


class _FakeSession:
    server = "python"

    def ensure_alive(self):
        pass


class _LeakyLedger:
    def __init__(self, gi):
        pass

    def cleanup(self):
        return ["cleanup history abc: boom"]


def _ctx():
    from types import SimpleNamespace

    return SimpleNamespace(
        gi=None,
        galaxy_url="http://g",
        api_key="k",
        version="26.0",
        user={},
        contract=None,
        results={},
    )


class TestCleanupFailures:
    def _scenario(self):
        from tests.live.model import Pack, Scenario

        scenario = Scenario.model_validate(
            {"id": "s.x", "note": "n", "steps": [{"call": "get_user"}]}
        )
        return Pack.model_validate({"pack": "p", "tools": []}), scenario

    def test_cleanup_failure_turns_an_xfail_into_a_failure(self, monkeypatch):
        from tests.live import runner

        def boom(*a, **k):
            raise ExpectedFailureError("known bug")

        monkeypatch.setattr(runner, "Ledger", _LeakyLedger)
        monkeypatch.setattr(runner, "_call", boom)
        pack, scenario = self._scenario()
        ctx = _ctx()
        with pytest.raises(AssertionError, match="cleanup left things"):
            runner.run_scenario(ctx, pack, scenario, _FakeSession())
        assert ctx.results[("s.x", "python")].stopped_by is None

    def test_cleanup_failure_fails_a_passing_run(self, monkeypatch):
        from tests.live import runner

        monkeypatch.setattr(runner, "Ledger", _LeakyLedger)
        monkeypatch.setattr(runner, "_call", lambda *a, **k: None)
        pack, scenario = self._scenario()
        with pytest.raises(AssertionError, match="cleanup left things"):
            runner.run_scenario(_ctx(), pack, scenario, _FakeSession())


class TestProductionGuard:
    def test_usegalaxy_is_refused_without_the_override(self):
        from tests.live.options import production_refusal

        assert production_refusal("https://usegalaxy.org", None)
        assert production_refusal("https://usegalaxy.eu/", "0")
        assert production_refusal("https://usegalaxy.org", "1") is None
        assert production_refusal("http://localhost:8091", None) is None

    def test_the_sweep_applies_it(self, monkeypatch, capsys):
        from tests.live import sweep

        monkeypatch.setenv("GALAXY_URL", "https://usegalaxy.org/")
        monkeypatch.setenv("GALAXY_API_KEY", "k")
        monkeypatch.delenv("GALAXY_LIVE_ALLOW_PRODUCTION", raising=False)
        monkeypatch.setattr(sweep, "sweep", lambda *a: pytest.fail("swept production"))
        assert sweep.main([]) == 2
        assert "production Galaxy" in capsys.readouterr().err


def test_tokens_one_run_bound_to_the_same_value_compare_equal():
    # An HDA and a history can share an encoded id; whichever was bound first names it.
    py = _run("python", [])
    py.bound = [("aaaaaaaaaaaaaaaa", "<h.id>"), ("bbbbbbbbbbbbbbbb", "<gamma>")]
    ts = _run("typescript", [])
    ts.bound = [("cccccccccccccccc", "<h.id>"), ("cccccccccccccccc", "<gamma>")]
    py.steps = [
        StepRecord(0, "t", {"outcome": "success", "data": {"o": "bbbbbbbbbbbbbbbb"}}, "exact", [])
    ]
    ts.steps = [
        StepRecord(0, "t", {"outcome": "success", "data": {"o": "cccccccccccccccc"}}, "exact", [])
    ]
    assert compare_runs(py, ts) == []
    ts.bound = [("cccccccccccccccc", "<h.id>"), ("dddddddddddddddd", "<gamma>")]
    assert compare_runs(py, ts) == [
        'step 0 t [exact] data.o: python "<gamma>", typescript "<h.id>"'
    ]


class TestDocumentedOutcome:
    """histories.create.empty_name: TS's documented answer is an argument rejection."""

    def _row(self):
        return _row(
            tool="create_history",
            where="outcome",
            kind="outcome-mismatch",
            server="typescript",
            adapter={"expect": "failure", "outcome": "failure:arguments"},
        )

    @staticmethod
    def _failure(kind, sentence):
        return {"outcome": "failure", "failure": {"kind": kind, "sentence": sentence}}

    def test_an_outcome_row_must_name_its_outcome(self):
        with pytest.raises(ValueError, match="names the outcome"):
            _row(where="outcome", kind="outcome-mismatch", adapter={"expect": "failure"})

    def test_the_documented_rejection_is_an_xfail(self):
        view = self._failure("arguments", "MCP error -32602: too_small")
        failures = ["outcome: expected success, got failure: MCP error -32602: too_small"]
        with pytest.raises(ExpectedFailureError):
            _conclude("w", "create_history", view, failures, [], self._row(), "typescript")

    def test_a_different_failure_kind_fails_the_step(self):
        view = self._failure("tool", "service unavailable")
        failures = ["outcome: expected success, got failure: service unavailable"]
        with pytest.raises(AssertionError, match="documents failure:arguments") as info:
            _conclude("w", "create_history", view, failures, [], self._row(), "typescript")
        assert not isinstance(info.value, ExpectedFailureError)

    def _runs(self, ts_view):
        py_view = {"outcome": "success", "data": {"name": "Unnamed history"}, "message": "m"}
        py = _run("python", [StepRecord(0, "create_history", py_view, "shape", [])])
        ts = _run(
            "typescript",
            [
                StepRecord(
                    0, "create_history", ts_view, "shape", [], ["outcome"], "failure:arguments"
                )
            ],
        )
        return py, ts

    def test_parity_accepts_the_documented_rejection(self):
        assert compare_runs(*self._runs(self._failure("arguments", "-32602"))) == []

    def test_parity_fails_a_different_failure_kind(self):
        problems = compare_runs(*self._runs(self._failure("tool", "service unavailable")))
        assert problems == [
            "step 0 create_history: typescript answered failure:tool, but its expect-failure "
            "row documents failure:arguments"
        ]


class TestAliasChains:
    A, B, C, H, X = (c * 16 for c in "abcde")

    def _pair(self, py_bound, ts_bound, py_value, ts_value):
        py = _run("python", [])
        ts = _run("typescript", [])
        py.bound, ts.bound = py_bound, ts_bound
        py.steps = [StepRecord(0, "t", {"outcome": "success", "o": py_value}, "exact", [])]
        ts.steps = [StepRecord(0, "t", {"outcome": "success", "o": ts_value}, "exact", [])]
        return py, ts

    def test_two_names_linked_only_through_a_shared_name_match(self):
        # python: h.id == gamma; typescript: c.id == gamma. Both answer gamma correctly, and
        # it masks to <h.id> on one side and <c.id> on the other.
        py, ts = self._pair(
            [(self.A, "<h.id>"), (self.A, "<gamma>"), (self.C, "<c.id>")],
            [(self.H, "<h.id>"), (self.B, "<c.id>"), (self.B, "<gamma>")],
            self.A,
            self.B,
        )
        assert compare_runs(py, ts) == []

    def test_a_genuine_id_mismatch_still_fails(self):
        # Same python run, but typescript's c.id is not gamma, and it answers c.id.
        py, ts = self._pair(
            [(self.A, "<h.id>"), (self.A, "<gamma>"), (self.C, "<c.id>")],
            [(self.H, "<h.id>"), (self.C, "<c.id>"), (self.X, "<gamma>")],
            self.A,
            self.C,
        )
        assert compare_runs(py, ts) == ['step 0 t [exact] o: python "<h.id>", typescript "<c.id>"']


class TestServerEnv:
    """A scenario's `server_env`: GALAXY_MCP_* only, references checked, a server of its own."""

    @staticmethod
    def _scenario(server_env: dict[str, str]) -> dict[str, object]:
        return {
            "id": "scope.env.case",
            "note": "a scenario with a server environment of its own",
            "fixtures": [{"as": "a", "history": {}}],
            "server_env": server_env,
            "steps": [{"call": "get_server_info"}],
        }

    def test_only_the_servers_own_settings_may_be_set(self):
        from tests.live.model import Scenario

        Scenario.model_validate(self._scenario({"GALAXY_MCP_SCOPE_WRITE_HISTORIES": "${a.id}"}))
        for name in ("GALAXY_URL", "GALAXY_API_KEY", "PATH", "GALAXY_LIVE_ALLOW_PRODUCTION"):
            with pytest.raises(ValueError, match="server_env may only set GALAXY_MCP_"):
                Scenario.model_validate(self._scenario({name: "x"}))

    def test_a_reference_to_nothing_is_an_offline_problem(self):
        from tests.live.loader import check_references
        from tests.live.model import Scenario

        good = Scenario.model_validate(
            self._scenario({"GALAXY_MCP_SCOPE_WRITE_HISTORIES": "${a.id}"})
        )
        assert check_references(good) == []
        bad = Scenario.model_validate(
            self._scenario({"GALAXY_MCP_SCOPE_WRITE_HISTORIES": "${nope.id}"})
        )
        assert any("server_env" in problem for problem in check_references(bad))

    def test_a_dedicated_session_is_not_the_shared_one_and_takes_the_extra_environment(
        self, tmp_path
    ):
        pool = servers.ServerPool(
            {}, "http://galaxy.invalid", "k", "/usr/bin", str(tmp_path), tmp_path
        )
        own = pool.dedicated("python")
        assert own is not pool.dedicated("python")
        assert pool._sessions == {}
        own.env.update({"GALAXY_MCP_SCOPE_CREATED_ONLY": "1"})
        assert own.env["GALAXY_URL"] == "http://galaxy.invalid"
        assert own.env["GALAXY_MCP_SCOPE_CREATED_ONLY"] == "1"
        assert own.starts == 0
