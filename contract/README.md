# contract

What the Python and TypeScript implementations are both held to. Neither side owns these
files: Python generates most of them, and both test suites read them. CI runs both suites
when anything here changes.

| Path | What it is | Written by |
| --- | --- | --- |
| `mcp-surface.json` | Every tool the Python server advertises: parameters, types, defaults, mutability, the Galaxy version it needs | `uv run python -m tests.surface_manifest` in `python/` |
| `envelopes/` | Golden results: a recorded Galaxy reply and the answer the Python server gave for it, success and failure, replayed through the TypeScript MCP server and CLI | `uv run python -m tests.envelope_fixtures` in `python/` |
| `galaxy-builtin-surface.json` | Snapshot of the MCP server Galaxy itself serves, stamped with the Galaxy commit | `python/tests/builtin_surface.py`, by hand against a Galaxy checkout |
| `accepted-divergences.json` | Every known difference between the surfaces, with a status and a reason | edited by hand |
| `python-unicode/` | Python's Unicode tables, so TypeScript can match Python's string handling exactly | `uv run python -m tests.python_unicode` in `python/` |

`PARITY.md` at the repo root is generated from these (`pnpm parity:report` in `typescript/`).

The contract can change. When it does, both implementations and these files change in the
same pull request.
