"""Write down the Unicode facts this interpreter's string methods are built on.

Three of this server's answers are decided by Unicode data rather than by any rule
either language spells out:

* ``\\w``, and through it ``\\b``, which is where ``_tokenize_for_search`` cuts an
  intent into terms -- a word character on one side and a separator on the other is
  the whole of what a boundary is;
* ``str.isspace()``, which is what ``str.strip()`` removes and what ``str.split()``
  splits on, used by ``_clean_readme_summary``;
* ``str.lower()``, which the three search tools lower both needle and haystack with
  before asking whether one contains the other.

The TypeScript surfaces have to answer the same way, and their runtime carries a
different edition of the data: Node 22 is on Unicode 17.0 and this interpreter reports
15.0.0, which knows 9,661 fewer code points as letters or numbers. Asking each runtime
its own tables therefore gives two answers to the same question -- an intent of
``rnaseq`` followed by a letter Unicode added after 15.0 has a term here and none
there. So the tables are not asked for at runtime. They are read off this interpreter
once, written down here, and the other side is built from this file.

The file is checked in with the Unicode version in its name because that version is the
contract, not an implementation detail: moving to an interpreter with newer tables
changes what the tools answer and has to arrive as a reviewable diff.

Regenerate with ``uv run python -m tests.python_unicode``;
``tests/test_python_unicode.py`` fails when the checked-in file is stale.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections.abc import Callable, Iterable
from pathlib import Path

# The interpreter the fixtures are generated with, and so the one whose Unicode data
# every surface follows. Changing this is a deliberate act: regenerate, look at what
# moved in the counts below, and say so in the CHANGELOG.
PINNED_UNICODE_VERSION = "15.0.0"

DATA_PATH = Path(__file__).parent / "testdata" / f"python-unicode-{PINNED_UNICODE_VERSION}.json"
REGENERATE_COMMAND = "uv run python -m tests.python_unicode"

# One past the last code point, which is what `chr` accepts up to.
CODE_SPACE = 0x110000

_WORD = re.compile(r"\w")


def ranges_where(predicate: Callable[[int], bool]) -> list[list[int]]:
    """Every code point the predicate holds for, as inclusive ranges."""
    out: list[list[int]] = []
    start: int | None = None
    for code_point in range(CODE_SPACE):
        if predicate(code_point):
            if start is None:
                start = code_point
        elif start is not None:
            out.append([start, code_point - 1])
            start = None
    if start is not None:
        out.append([start, CODE_SPACE - 1])
    return out


def word_char_ranges() -> list[list[int]]:
    """The set ``\\w`` matches on a ``str``, which is the set ``\\b`` reads.

    Asked of the engine rather than assembled out of general categories, so what is
    written down is what the pattern does and not a second opinion about it.
    """
    return ranges_where(lambda code_point: _WORD.match(chr(code_point)) is not None)


def whitespace_ranges() -> list[list[int]]:
    """The set ``str.isspace()`` is true for -- ``strip()``'s and ``split()``'s."""
    return ranges_where(lambda code_point: chr(code_point).isspace())


def lower_mapping() -> list[list[object]]:
    """Every code point ``str.lower()`` does not leave alone, and what it makes of it.

    One code point at a time, so the context-sensitive part of lowercasing (a Greek
    capital sigma at the end of a word) is not in here -- that one is a rule, not a
    table, and both languages apply it.
    """
    out: list[list[object]] = []
    for code_point in range(CODE_SPACE):
        character = chr(code_point)
        lowered = character.lower()
        if lowered != character:
            out.append([code_point, lowered])
    return out


def _render_rows(rows: Iterable[list[object]]) -> str:
    """One row per line, so a table of thousands of entries still reviews as a diff."""
    return "\n".join(f"      {json.dumps(row, separators=(', ', ': '))}," for row in rows)


def build() -> str:
    """The file's text, exactly as it is checked in."""
    words = word_char_ranges()
    spaces = whitespace_ranges()
    lowers = lower_mapping()

    def table(name: str, rows: list[list[object]], counted: str, last: bool = False) -> str:
        lines = _render_rows(rows)
        # No trailing comma on the last row: this is JSON, not a Python literal.
        lines = lines[:-1] if lines.endswith(",") else lines
        return (
            f'  "{name}": {{\n'
            f'    "count": {len(rows)},\n'
            f"{counted}"
            f'    "rows": [\n{lines}\n    ]\n'
            f"  }}{'' if last else ','}\n"
        )

    def code_points(rows: list[list[int]]) -> str:
        total = sum(high - low + 1 for low, high in rows)
        return f'    "code_points": {total},\n'

    return (
        "{\n"
        f'  "unicode_version": {json.dumps(PINNED_UNICODE_VERSION)},\n'
        f'  "regenerate": {json.dumps(REGENERATE_COMMAND)},\n'
        '  "note": "Inclusive code point ranges, low first, except lower_map, which is '
        "[code point, str.lower() of it]. Read off the interpreter named above; the "
        'TypeScript side is built from this file rather than from its own tables.",\n'
        + table("word_chars", list(words), code_points(words))
        + table("whitespace", list(spaces), code_points(spaces))
        + table("lower_map", lowers, "", last=True)
        + "}\n"
    )


def main() -> int:
    if unicodedata.unidata_version != PINNED_UNICODE_VERSION:
        print(
            f"this interpreter carries Unicode {unicodedata.unidata_version} and the pinned "
            f"tables are {PINNED_UNICODE_VERSION}; regenerating here would change what the "
            "tools answer, so bump PINNED_UNICODE_VERSION in tests/python_unicode.py "
            "deliberately (and say so in the CHANGELOG) if that is the move",
            file=sys.stderr,
        )
        return 1
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(build())
    print(f"wrote {DATA_PATH.relative_to(Path(__file__).parent.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
