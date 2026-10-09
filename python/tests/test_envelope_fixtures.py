"""The checked-in golden envelopes still describe what the tools return.

The files under ``contract/envelopes`` are the contract the TypeScript MCP server and
CLI are held to. They are only worth anything while they say what this server actually
emits, so this regenerates every case and compares. A tool whose envelope changes fails
here first -- before the other language's replay -- with the command to regenerate.
"""

from __future__ import annotations

import pytest

from .envelope_fixtures import FIXTURE_ROOT, REGENERATE_COMMAND, build


@pytest.fixture(scope="module")
def generated() -> dict[str, str]:
    return build()


def test_every_case_is_written_down(generated: dict[str, str]) -> None:
    missing = sorted(name for name in generated if not (FIXTURE_ROOT / name).exists())
    assert not missing, (
        f"no golden file for {', '.join(missing)}; run `{REGENERATE_COMMAND}` from python/"
    )


def test_nothing_stale_is_left_behind(generated: dict[str, str]) -> None:
    on_disk = {str(path.relative_to(FIXTURE_ROOT)) for path in FIXTURE_ROOT.rglob("*.json")}
    extra = sorted(on_disk - set(generated))
    assert not extra, (
        f"{', '.join(extra)} belongs to no case any more; run `{REGENERATE_COMMAND}` from python/"
    )


def test_the_envelopes_still_match(generated: dict[str, str]) -> None:
    stale = []
    for name, text in sorted(generated.items()):
        path = FIXTURE_ROOT / name
        if path.exists() and path.read_text() != text:
            stale.append(name)
    assert not stale, (
        f"{', '.join(stale)} no longer match what the server returns; run "
        f"`{REGENERATE_COMMAND}` from python/ and review the diff -- the "
        "TypeScript surfaces are compared against these files"
    )
