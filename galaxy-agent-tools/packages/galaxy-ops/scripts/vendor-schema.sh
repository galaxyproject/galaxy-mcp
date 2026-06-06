#!/usr/bin/env bash
# Regenerate src/generated/schema.ts from local Galaxy 26.1 source.
# Swap target: @galaxyproject/galaxy-api-client@26.1.x once it publishes to npm.
set -euo pipefail
GALAXY="${GALAXY_SRC:-$HOME/work/galaxy}"
PKG_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$PKG_ROOT/src/generated/schema.ts"
# PID-suffixed temp (portable: BSD mktemp won't substitute X's before a suffix).
TMP="${TMPDIR:-/tmp}/galaxy_schema_$$.yaml"
trap 'rm -f "$TMP"' EXIT

# 1) Dump the OpenAPI spec from local source (Galaxy's own pipeline; Makefile:191).
( cd "$GALAXY" && . .venv/bin/activate && python scripts/dump_openapi_schema.py "$TMP" )

# 2) Generate TS types with the same generator Galaxy uses (openapi-typescript ^7).
# Run from the package root so pnpm exec resolves devDeps from packages/galaxy-ops.
# We intentionally skip Galaxy's prettier pass: openapi-typescript's output is already
# clean and deterministic (regen is byte-identical), so prettier would only add a devDep
# and risk a whole-file reformat diff for a generated artifact nobody hand-edits.
( cd "$PKG_ROOT" && pnpm exec openapi-typescript "$TMP" -o "$OUT" )
echo "Wrote $OUT"
