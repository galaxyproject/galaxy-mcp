# contract

What the Python and TypeScript implementations are both held to. Neither side owns these
files: Python generates most of them, and both test suites read them. CI runs both suites
when anything here changes.

| Path | What it is | Written by |
| --- | --- | --- |
| `mcp-surface.json` | Every tool the Python server advertises: parameters, types, defaults, mutability, the Galaxy version it needs | `uv run python -m tests.surface_manifest` in `python/` |
| `envelopes/` | Golden results: a recorded Galaxy reply and the answer the Python server gave for it, success and failure, replayed through the TypeScript MCP server and CLI | `uv run python -m tests.envelope_fixtures` in `python/` |
| `galaxy-builtin-surface.json` | Snapshot of the MCP server Galaxy itself serves, stamped with the Galaxy commit | `python/tests/builtin_surface.py`, by hand against a Galaxy checkout |
| `accepted-divergences.json` | Every known difference between the surfaces, with a status and a reason. Its `live` section holds the differences only a live run sees, each a one-sided adapter the live scenarios cite | edited by hand |
| `scenarios/<pack>.json` | Live scenarios: tool calls and assertions run against a real Galaxy through both MCP servers over stdio, then compared across them | edited by hand; run with `uv run pytest tests/live -m live --no-cov` in `python/` (see `python/tests/live/__init__.py`); in CI by `.github/workflows/live-scenarios.yml`, manual-only until a test Galaxy exists |
| `scenarios/data/<pack>/` | Files a pack's fixtures load, such as workflow `.ga` files | edited by hand |
| `scenarios/scenario.schema.json` | JSON Schema for a scenario pack | `uv run python -m tests.live.model` in `python/` |
| `python-unicode/` | Python's Unicode tables, so TypeScript can match Python's string handling exactly | `uv run python -m tests.python_unicode` in `python/` |

`PARITY.md` at the repo root is generated from these (`pnpm parity:report` in `typescript/`).

Envelopes are recorded on the handshake protocol era (`mode="legacy"` in
`python/tests/mcp_session.py`). On the sessionless 2026-07-28 era the Python server's results
also carry `_meta.serverInfo` and `resultType`, which the TypeScript server does not send.

The contract can change. When it does, both implementations and these files change in the
same pull request.
