"""Live scenarios: one language-neutral script, run against both MCP servers (#119).

Scenario packs live in contract/scenarios/<pack>.json. Each scenario runs once per
server, each server a real stdio subprocess (`python -m galaxy_mcp` and
`node typescript/packages/galaxy-mcp/dist/index.js`) with only PATH, HOME, GALAXY_URL and
GALAXY_API_KEY in its environment. Results are normalised identically for both, checked
against the scenario's assertions, then compared across servers.

Running (from python/, with GALAXY_URL and GALAXY_API_KEY exported):

    (cd ../typescript && pnpm -r build)
    uv run pytest tests/live -m live --no-cov                          # everything
    uv run pytest tests/live -m live --no-cov --live-pack smoke        # one pack, both servers
    uv run pytest tests/live -m live --no-cov --live-server python      # one server
    uv run pytest tests/live -m live --no-cov -k smoke.history         # by scenario id
    uv run pytest tests/live -m live --no-cov --live-waive network     # skip, not fail
    uv run pytest tests/test_live_scenario_files.py                    # offline checks only
    uv run python -m tests.live.sweep --dry-run                        # crashed runs' leftovers

A plain `uv run pytest` deselects the live tests and never needs Galaxy. Once a live test
is selected, a preflight stops the run (exit code 4) on a missing or unreachable Galaxy,
a rejected key, a stale TS build, scenario-file problems, or the old GALAXY_TEST_URL /
GALAXY_TEST_API_KEY names. GALAXY_LIVE_PYTHON_CMD / GALAXY_LIVE_TS_CMD replace a server
command (shlex-split); a usegalaxy.* URL is refused unless GALAXY_LIVE_ALLOW_PRODUCTION=1.

Tests produced: test_surface[<server>] (tools/list against the manifest),
test_scenario[<server>-<id>], test_parity[<id>] (both servers' results after masking;
deselected under --live-server). Serial only.

Adding a scenario -- the format is model.py, the JSON Schema is
contract/scenarios/scenario.schema.json (regenerate with `uv run python -m tests.live.model`):

    {"id": "area.thing.case", "note": "why it exists",
     "requires": {"tools": ["cat1"], "capabilities": ["network"]},
     "fixtures": [{"as": "h", "history": {}},
                  {"as": "d", "dataset": {"history": "${h.id}", "content": "a\\n"}}],
     "steps": [
       {"call": "get_dataset_details", "args": {"dataset_id": "${d.id}"},
        "expect": [{"path": "data.state", "equals": "ok"}], "save": {"ds": "data.id"}},
       {"wait": {"dataset": "${ds}", "state": "ok"}},
       {"verify": {"dataset_text": "${ds}"}, "expect": [{"path": "content", "equals": "a\\n"}]}]}

- Fixtures are made through a separate bioblend client (never the server under test),
  once per server x scenario, and always cleaned up. Kinds and what they bind: history
  (id, name), dataset (id, name, hid, history_id), collection (id, name, hid,
  element_count), workflow (id, name; file under contract/scenarios/data/<pack>/),
  invocation (id), page (id, slug, title), user_tool (uuid, id), local_file (path, name,
  content, size), temp_path (path).
- References: "${a.b.0.c}". A string that is exactly one reference keeps the value's
  type. Builtins: run.tag (put it in every name you create), run.server, run.tmp,
  galaxy.url (GALAXY_URL as handed to both servers, trailing slash removed),
  galaxy.api_key (masked as <api_key> in parity and in failure messages),
  galaxy.version, galaxy.user.id/email/username. gt/gte/lt/lte take a number,
  {"path": ...} in the same view, or a whole reference to an earlier value.
- Steps: call (args, outcome, expect, save, creates, until, parity, mask, divergences),
  wait (dataset/job/invocation to a state, "terminal", or a list), verify (Galaxy's own
  JSON for history/dataset/workflow/invocation/page, or {content, size} for dataset_text
  and {exists, content, size} for file).
- `server_env` on a scenario adds GALAXY_MCP_* variables to both servers' environment,
  for behaviour configured at startup (the scope). Values may reference fixtures; such a
  scenario runs against a server process of its own, started after its fixtures.
- Anything a tool creates must be declared in `creates` so cleanup finds it.
- The call-step view is {outcome, data, success, message, count, pagination} or
  {outcome: "failure", failure: {kind: "tool"|"arguments", sentence}}. Assert
  failure.kind, never the sentence, for argument-validation failures.
- Other runs share the Galaxy user: never assert a global count.
- A Python/TS difference is either fixed in the server (separate PR) or recorded as a row
  in accepted-divergences.json -> live, status unreviewed-gap, and cited as
  "live/<tool> :: <where> :: <kind>" on the step (or scenario). Rows are one-sided
  adapters (drop, rstrip, sub, cut_from, set -- the one op for a non-string value, which
  swaps only a value equal to its `from` for its `to` -- or {"expect": "failure"} for a
  known server bug); an adapter that stops changing anything fails the run as stale.
  Recount live.ratchet.unreviewedGaps when you add one -- the offline check prints the
  number.
- A key has no server in it, so a bug both servers share is two rows (one per `server`)
  under one key, and one citation applies whichever row is about the server under test.
  When one tool has two unrelated divergences at the same path and kind, qualify the
  path in brackets -- "failure.sentence[preflight-src]" -- so each gets its own row and
  ruling; `where` is documentation, the adapter names the paths it touches.
- Static citations ("divergences/<tool> :: <param> :: <kind>") ignore runs of spaces, so
  a row with no param is "divergences/connect :: :: missing-ts-tool".

Details that are easy to miss:

- A collection fixture's elements are {name, content, ext?}; for list:paired each element
  is {name, elements: [{name: "forward", ...}, {name: "reverse", ...}]}. A paired
  collection has exactly the elements forward and reverse.
- A workflow fixture is imported under "${run.tag}-<name in the .ga>", so list filters by
  ${run.tag} find it. Key invocation inputs by step index.
- Galaxy moves an invocation on from scheduled to completed once its jobs finish, so a
  wait for "scheduled" also accepts "completed"; a verify or assertion on the state
  should accept both too.
- The preflight checks only the selected packs' scenarios, so a pack with an unfinished
  citation does not stop another pack's run; the offline check covers them all.
- Verify-step citations refer to the tool of the call step before it. An
  {"expect": "failure"} row must be cited on the step that fails, and the scenario stops
  there on that server (reported as xfail); parity compares the steps both servers ran,
  the failing one included with only the row's `where` left out of the comparison.
  The row only explains failures at its own `where` (or below it; `file` covers a whole
  verify:file view): an outcome flip, a missing `creates` or a failing assertion anywhere
  else on that step is a plain failure, so a new bug cannot hide behind a known one.
  A row at `outcome` names the outcome it documents ({"expect": "failure", "outcome":
  "failure:arguments"}); any other outcome or failure kind fails the step and parity.
  Cleanup that fails after an expected failure still fails the test.
- Scenario-level citations apply to every step calling the row's tool and are stale only
  if they change nothing in the whole run; step-level ones are stale per step.
- `parity: shape` (the default) compares the outcome, failure sentences, the success
  `message`, and the key structure with JSON types; a null only matches a null, but an
  empty list (or a null list element) matches any list. `parity: exact` compares masked
  values, so use it where the answers should be identical; `mask` patterns must match on
  both servers or the parity test fails, and should only cover values that really vary.
- The smoke pack owns no tools; the four packs in loader.EXPECTED_PACKS own the manifest
  between them, and the offline check fails if any of the four is missing.
- A plain run deselects live tests unless the -m expression names `live`, so another -m
  (say `-m "not slow"`) cannot run them against an exported GALAXY_URL by accident.
"""
