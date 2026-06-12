---
name: release
description: Use when cutting a release of galaxyproject/galaxy-mcp -- bumping the version and publishing the galaxy-mcp package to PyPI. Triggers include "cut a galaxy-mcp release", "release galaxy-mcp", "ship galaxy-mcp X.Y.Z", or /release.
---

# Release (galaxy-mcp)

## Overview

Cut a galaxy-mcp release end to end. The package publishes to **PyPI**, which is
**irreversible** -- a version string can never be re-uploaded. The trigger is
**publishing the GitHub release**: `python-release.yml` runs on `release: published`,
builds, and `twine upload`s to PyPI. Two actions are outward-facing and gated -- the
**push of the version-bump commit to `galaxyproject/main`** and the **release publish**
(the PyPI trigger). STOP for an explicit human "go" before each. Everything before the
push is local and reversible.

Why use this over hand-running commands: it gets the version right (the `.dev0` marker on
`main` is the source of truth, NOT release-drafter's auto-guess), it uses `uv lock` for
the lockfile instead of a foot-gun `sed`, and it gates the publish so nothing ships on
autopilot.

**Violating the letter of the gates is violating the spirit.** Don't collapse the flow
into one script. Stop, show, and wait for a human "go" before each publish boundary.

## Layout / facts (easy to get wrong)

- The Python package lives in **`mcp-server-galaxy-py/`**, not the repo root. Three files
  carry the version: `pyproject.toml`, `src/galaxy_mcp/__init__.py`, and `uv.lock`.
- `main` between releases carries a dev version like `X.Y.0.dev0`. **The release version
  = that minus `.dev0`** (e.g. `1.7.0.dev0` -> `1.7.0`). The dev marker is the intent.
- Releases are cut from `galaxyproject/main`. The primary checkout is often on a feature
  branch -- **always work from a clean worktree off canonical main**, never in place.
- `release-drafter` keeps a DRAFT release, but it resolves the version from PR labels and
  **under-resolves to a patch when PRs lack a `feature`/`minor` label** (it drafted
  `v1.6.1` when the real release was `1.7.0`). Trust the `.dev0` marker, not the draft tag.

## Modes

- `/release` -- full guided run.
- `/release preview` -- Phase 0-1 only: determine version + gather notes, then STOP. Writes/pushes nothing.
- `/release <version>` -- pin the version (e.g. `1.7.0`) instead of deriving it.

## Phase 0 -- Preconditions + clean worktree

Abort cleanly on any failure, with a specific message.

```bash
gh auth status
# Canonical remote = whichever remote pushes to galaxyproject/galaxy-mcp (don't assume "upstream").
# Use grep, NOT awk: a `[:/]` bracket in awk is read as a POSIX class on macOS/BSD awk and aborts
# with "nonterminated character class", leaving CANON empty and cascading into `/main` errors later.
CANON=$(git remote -v | grep '(push)' | grep -E 'galaxyproject/galaxy-mcp(\.git)? ' | head -1 | awk '{print $1}')
test -n "$CANON" || { echo "ERROR: no remote points at galaxyproject/galaxy-mcp -- add one and retry"; exit 1; }
git fetch "$CANON" --tags
# main must be green before releasing:
gh run list --repo galaxyproject/galaxy-mcp --branch main --workflow "Python Tests" -L 1 \
  --json conclusion -q '.[0].conclusion'        # expect: success
# Author from a clean worktree off canonical main (leaves your checkout untouched):
git worktree add -b rel-tmp ../gmcp-release "$CANON/main"
cd ../gmcp-release/mcp-server-galaxy-py
```

## Phase 1 -- Version + notes

```bash
CUR=$(grep -E '^version ?=' pyproject.toml | head -1 | sed -E 's/.*"([^"]+)".*/\1/')   # e.g. 1.7.0.dev0
LAST_TAG=$(git -C .. describe --tags --match 'v*' --abbrev=0)
```

- Release version = `$CUR` minus `.dev0` (or the `/release <version>` arg).
- Pull the drafted notes and the PRs since the last tag:

```bash
gh release list --repo galaxyproject/galaxy-mcp -L 5      # find the release-drafter DRAFT
gh release view <draft-tag> --repo galaxyproject/galaxy-mcp
gh pr list --repo galaxyproject/galaxy-mcp --state merged --base main \
  --search "merged:>$(git -C .. log -1 --format=%cs $LAST_TAG)" --json number,title,author,labels
```

Curate per the **Editorial rules** below. **Present the version + notes for human
edit/approval.** Nothing is written yet. `/release preview` ends here.

## Phase 2 -- Safe local prep (after approval)

```bash
NEW=1.7.0   # the approved version
sed -i.bak '/^\[project\]/,/^\[/ s/^version = "[^"]*"/version = "'"$NEW"'"/' pyproject.toml && rm -f pyproject.toml.bak
sed -i.bak 's/^__version__ = "[^"]*"/__version__ = "'"$NEW"'"/' src/galaxy_mcp/__init__.py && rm -f src/galaxy_mcp/__init__.py.bak
uv lock     # regenerates the galaxy-mcp self-version in uv.lock. NEVER sed uv.lock --
            # a bare s/1.7.0/.../ also rewrites other deps pinned at that version (e.g. frozenlist).
```

Verify, then test + build (mirrors CI and the deploy job exactly):

```bash
grep -E '^version ?=' pyproject.toml; grep __version__ src/galaxy_mcp/__init__.py
uv sync --all-extras && uv run pytest -q              # expect all green
uv run python -m build && uv run twine check dist/*   # both must PASS
```

Commit (do NOT push yet):

```bash
git add pyproject.toml src/galaxy_mcp/__init__.py uv.lock
git commit -m "Bump version to $NEW for release"
git --no-pager show --stat HEAD                       # show the 3-file diff
```

Reversible: `git reset --hard HEAD~1` undoes everything so far.

## Phase 3 -- GATE 1: push bump to main  (STOP)

⛔ **Hard stop.** Pushing puts `galaxyproject/main` on the release version. Show the diff;
wait for an explicit "go". On go (fast-forward only -- re-check main hasn't moved):

```bash
git fetch "$CANON" --quiet
[ "$(git rev-parse $CANON/main)" = "$(git rev-parse HEAD~1)" ] || { echo "main moved -- re-sync before pushing"; exit 1; }
git push "$CANON" HEAD:main
```

## Phase 4 -- Curated release body

Write the approved notes to a file. Don't delete the release-drafter draft and don't
hard-code its tag: release-drafter recomputes the *same* draft object on every push to
main, so its tag drifts between when you start the cut and when you publish (seen live: a
`v1.7.1` draft was renamed to `v1.8.1` mid-cut). At Gate 2 -- in the same shell as the
publish, so it can't go stale across the stop -- we resolve the draft tag fresh, then
retag and publish that existing draft in place. No delete, which also sidesteps a
permission classifier that (rightly) won't auto-approve deleting a release the agent
didn't create.

```bash
printf '%s\n' "$APPROVED_NOTES" > /tmp/gmcp-notes.md   # the notes FILE persists across the Gate 2 stop; the draft tag is resolved fresh at the gate (below)
```

## Phase 5 -- GATE 2: publish release -> PyPI  (STOP)

⛔ **Hard stop.** Publishing fires `python-release.yml`, which **uploads
`galaxy-mcp $NEW` to PyPI -- this cannot be undone.** Main must already carry `$NEW`
(Gate 1). Wait for an explicit "go". The "go" authorizes a single publish action -- one
command, no bundled delete (a bundled `delete && create` gets the whole line denied). On go:

```bash
# Resolve the draft tag fresh HERE -- same shell as the publish, so drift across the gate
# can't bite and the var is guaranteed set. Never a literal like v1.7.1. Empty if no draft.
DRAFT_TAG=$(gh release list --repo galaxyproject/galaxy-mcp \
  --json tagName,isDraft -q '[.[]|select(.isDraft)][0].tagName')
if [ -n "$DRAFT_TAG" ]; then
  # Retag the existing draft to v$NEW and publish it in place -- applies our curated notes,
  # creates tag v$NEW at main, fires the deploy. No delete needed.
  gh release edit "$DRAFT_TAG" --repo galaxyproject/galaxy-mcp \
    --tag "v$NEW" --target main --title "v$NEW" --notes-file /tmp/gmcp-notes.md --draft=false
else
  # No draft existed -- create the release fresh.
  gh release create "v$NEW" --repo galaxyproject/galaxy-mcp \
    --target main --title "v$NEW" --notes-file /tmp/gmcp-notes.md
fi
```

Watch the deploy; STOP and report if it fails. The run may not exist the instant publish
returns, so poll for it -- **filtered to this release's tag** (`--branch "v$NEW"`). Without
that filter a bare `-L 1` returns the *previous* release's already-green run instantly, the
loop breaks on iteration 1, and you'd watch the wrong (passing) run while the real deploy is
still building:

```bash
for i in $(seq 1 10); do
  RUN=$(gh run list --repo galaxyproject/galaxy-mcp --workflow "Release Python Package" \
    --event release --branch "v$NEW" -L 1 --json databaseId -q '.[0].databaseId')
  [ -n "$RUN" ] && break
  sleep 3
done
test -n "$RUN" || { echo "deploy run not found yet -- check the Actions tab"; exit 1; }
gh run watch --repo galaxyproject/galaxy-mcp "$RUN" --exit-status
# Confirm it actually landed on PyPI:
curl -s https://pypi.org/pypi/galaxy-mcp/json | python3 -c "import sys,json;print(json.load(sys.stdin)['info']['version'])"   # expect $NEW
```

## Phase 6 -- Post-release dev bump

Set `main` back to the next-minor dev version so the next cut's `.dev0` marker is correct
(this is what makes Phase 1's version derivation work).

```bash
DEV=1.8.0.dev0   # next minor + .dev0
sed -i.bak '/^\[project\]/,/^\[/ s/^version = "[^"]*"/version = "'"$DEV"'"/' pyproject.toml && rm -f pyproject.toml.bak
sed -i.bak 's/^__version__ = "[^"]*"/__version__ = "'"$DEV"'"/' src/galaxy_mcp/__init__.py && rm -f src/galaxy_mcp/__init__.py.bak
uv lock
git add pyproject.toml src/galaxy_mcp/__init__.py uv.lock
git commit -m "Bump version to $DEV for development"
git push "$CANON" HEAD:main
```

Clean up (from the primary checkout): `git worktree remove ../gmcp-release && git branch -D rel-tmp`.

## Editorial rules for notes

- Order by user impact: features first, then fixes, then CI/maintenance. Group related PRs.
- Keep release-drafter's `- $TITLE @author (#N)` line format -- maintainers expect it.
- Plain hyphens, never a Unicode em-dash. Keep a `## Contributors` line if the draft had one.
- Notes are public copy: ALWAYS show them for human approval before publishing.

## Red flags -- STOP

- About to push the bump or publish the release without an explicit "go" this turn -> STOP, show, wait.
- About to push to your fork (`origin`) instead of `$CANON` (galaxyproject) -> wrong remote.
- Took release-drafter's patch version (e.g. `v1.6.1`) when `main` says `1.7.0.dev0` -> use the `.dev0` marker; the draft mislabeled it.
- Hard-coded the draft tag (e.g. `v1.7.1`) or bundled `delete && create` for the publish -> the tag drifts and a bundled delete gets the whole line denied. Look the draft tag up fresh and retag-publish in place.
- `sed`-ed the version in `uv.lock` -> it also rewrote other deps pinned at that version. Use `uv lock`.
- Cutting in the primary checkout (maybe a feature branch) instead of a worktree off `$CANON/main`.
- Deploy failed but moving on -> never assume PyPI got it; verify via the PyPI JSON.
- Skipped `__init__.py` or `uv.lock` -> all three version files move together.

## Common mistakes

| Mistake | Fix |
|---|---|
| Wrong version (drafter's patch) | Drop `.dev0` from `main`'s version; that's the intent |
| Hard-coded / deleted the drafter draft | Look up the draft tag fresh; retag-publish it in place (`gh release edit --draft=false`) |
| `sed` on `uv.lock` | `uv lock` regenerates only the self-version line |
| Forgot `__init__.py` / `uv.lock` | Bump all three version files together |
| Pushed to the fork | Push to `$CANON` (galaxyproject/galaxy-mcp) |
| Published before tests/build | `uv run pytest` + `python -m build` + `twine check` locally first |
| Re-cut after a bad upload | PyPI can't replace a version -- bump to a new one, never reuse |
| Skipped the post-release dev bump | `main` must carry the next `.dev0` for the next cut to work |
