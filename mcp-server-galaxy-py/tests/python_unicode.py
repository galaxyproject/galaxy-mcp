"""Write down the Unicode facts this interpreter's string methods are built on.

Four of this server's answers are decided by Unicode data rather than by any rule
either language spells out:

* ``\\w``, and through it ``\\b``, which is where ``_tokenize_for_search`` cuts an
  intent into terms -- a word character on one side and a separator on the other is
  the whole of what a boundary is;
* ``str.isspace()``, which is what ``str.strip()`` removes and what ``str.split()``
  splits on, used by ``_clean_readme_summary``;
* ``str.lower()``'s one-code-point mapping, which the three search tools lower both
  needle and haystack with before asking whether one contains the other;
* ``str.lower()``'s one context-sensitive rule -- the final sigma -- which reads two
  further sets of its own, the cased characters and the case-ignorable ones.

The TypeScript surfaces have to answer the same way, and their runtime carries a
different edition of the data: Node 22 is on Unicode 17.0 and this interpreter reports
15.0.0, which knows 9,661 fewer code points as letters or numbers. Asking each runtime
its own tables therefore gives two answers to the same question -- an intent of
``rnaseq`` followed by a letter Unicode added after 15.0 has a term here and none
there, and a sigma after a letter Unicode gave a case to after 15.0 is a final sigma
there and an ordinary one here. So the tables are not asked for at runtime. They are
read off this interpreter once, written down here, and the other side is built from
this file.

The file is checked in with the Unicode version in its name because that version is the
contract, not an implementation detail: moving to an interpreter with newer tables
changes what the tools answer and has to arrive as a reviewable diff.

The final sigma rule, as this interpreter applies it
----------------------------------------------------
``str.lower()`` sends every code point through the one-code-point table -- except
U+03A3 GREEK CAPITAL LETTER SIGMA, the one character whose lowercase depends on what is
around it. In ``Objects/unicodeobject.c``, ``lower_ucs4`` hands ``0x3A3`` to
``handle_capital_sigma`` and everything else to ``_PyUnicode_ToLowerFull``, and the
comment on that function gives the context as

    \\p{cased} \\p{case-ignorable}* U+03A3 !( \\p{case-ignorable}* \\p{cased} )

which its two loops read like this:

1. walk backwards from the sigma for as long as the character is case-ignorable
   (``_PyUnicode_IsCaseIgnorable``). If the walk runs off the start of the string the
   sigma is not final.
2. Otherwise the sigma is final so far when the character the walk stopped on is cased
   (``_PyUnicode_IsCased``). Note the order: case-ignorable is asked first, so a
   character that is both -- U+0345 COMBINING GREEK YPOGEGRAMMENI among them -- is
   walked past in step 1 and never asked whether it is cased.
3. Then walk forwards from the sigma the same way. The sigma stays final when that walk
   runs off the end of the string, and otherwise when the character it stopped on is
   not cased.
4. A final sigma is U+03C2 and anything else is U+03C3.

Both walks read the original string rather than the lowercased one, and no other
character in ``str.lower()`` looks at its neighbours at all.

The two sets are derived from the interpreter rather than from ``unicodedata``, the same
way ``\\w`` is -- what is written down is what ``lower()`` does, not a second opinion
about it -- by asking it to lowercase a sigma in three shapes:

* ``(c + "\\u03a3").lower()`` ends in a final sigma exactly when ``c`` is cased: were
  ``c`` case-ignorable, step 1 would walk past it and off the start of the string.
  Because step 1 comes first, the set written down here is "cased and not
  case-ignorable", which is exactly the question step 2 asks.
* ``("A" + c + "\\u03a3").lower()`` ends in a final sigma when ``c`` is case-ignorable
  (the ``A`` is cased, so the walk reaching it means it walked over ``c``) or when ``c``
  is itself cased, so the case-ignorable set is that answer minus the cased ones.
* the same two questions from the other side, ``("A\\u03a3" + c)`` and
  ``("A\\u03a3" + c + "A")``, have to name the same two sets. ``build()`` asserts it.

``lower_by_the_tables`` below applies steps 1-4 using nothing but the three tables in
this file, and the generator checks it against the interpreter over every code point in
the space in each of the four probe shapes the TypeScript side tests. That check is what
makes the TypeScript ``pyLower`` a transcription rather than a guess.

Regenerate with ``uv run python -m tests.python_unicode``;
``tests/test_python_unicode.py`` fails when the checked-in files are stale.
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

TESTDATA = Path(__file__).parent / "testdata"
DATA_PATH = TESTDATA / f"python-unicode-{PINNED_UNICODE_VERSION}.json"
PROBE_PATH = TESTDATA / f"python-lower-probes-{PINNED_UNICODE_VERSION}.json"
REGENERATE_COMMAND = "uv run python -m tests.python_unicode"

# One past the last code point, which is what `chr` accepts up to.
CODE_SPACE = 0x110000

CAPITAL_SIGMA = "\u03a3"
FINAL_SIGMA = "\u03c2"
SMALL_SIGMA = "\u03c3"

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


def is_cased(code_point: int) -> bool:
    """Does the final-sigma rule's step 2 count this code point as a cased letter?

    A sigma immediately after it comes out final exactly when it does -- see the rule in
    the module docstring.
    """
    return (chr(code_point) + CAPITAL_SIGMA).lower().endswith(FINAL_SIGMA)


def is_case_ignorable(code_point: int) -> bool:
    """Does the rule's step 1 walk straight past this code point?

    Asked with a cased letter in front, so a final sigma means the backwards walk got
    all the way to the ``A``. Cased characters answer yes to that too and are taken out
    here, which is the same precedence the interpreter's two loops have.
    """
    if is_cased(code_point):
        return False
    return ("A" + chr(code_point) + CAPITAL_SIGMA).lower().endswith(FINAL_SIGMA)


def cased_ranges() -> list[list[int]]:
    return ranges_where(is_cased)


def case_ignorable_ranges() -> list[list[int]]:
    return ranges_where(is_case_ignorable)


def lower_mapping() -> list[list[object]]:
    """Every code point ``str.lower()`` does not leave alone, as a list of code points.

    Code points rather than a string because the other side walks code points too, and
    because a mapping is allowed to grow: U+0130 lowercases to two of them. One code
    point at a time, so the sigma is in here as its plain lowercase and the rule that
    overrides it lives in the two tables above.
    """
    out: list[list[object]] = []
    for code_point in range(CODE_SPACE):
        character = chr(code_point)
        lowered = character.lower()
        if lowered != character:
            out.append([code_point, [ord(c) for c in lowered]])
    return out


class Tables:
    """The written-down tables, in the shape the rule below reads them."""

    def __init__(
        self,
        lower_map: list[list[object]],
        cased: list[list[int]],
        case_ignorable: list[list[int]],
    ) -> None:
        self.lower = {
            int(row[0]): "".join(chr(c) for c in row[1])  # type: ignore[union-attr]
            for row in lower_map
        }
        self.cased = {cp for low, high in cased for cp in range(low, high + 1)}
        self.case_ignorable = {cp for low, high in case_ignorable for cp in range(low, high + 1)}


def lower_by_the_tables(text: str, tables: Tables) -> str:
    """``str.lower()``, out of the tables in this file and nothing else.

    Steps 1-4 of the module docstring, which is also what the TypeScript ``pyLower``
    does. Kept here so the transcription can be checked against the interpreter over the
    whole code space before the other side is built from the same tables.
    """
    out: list[str] = []
    length = len(text)
    for index, character in enumerate(text):
        if character != CAPITAL_SIGMA:
            out.append(tables.lower.get(ord(character), character))
            continue
        before = index - 1
        while before >= 0 and ord(text[before]) in tables.case_ignorable:
            before -= 1
        final = before >= 0 and ord(text[before]) in tables.cased
        if final:
            after = index + 1
            while after < length and ord(text[after]) in tables.case_ignorable:
                after += 1
            final = after == length or ord(text[after]) not in tables.cased
        out.append(FINAL_SIGMA if final else SMALL_SIGMA)
    return "".join(out)


# The four shapes the other side's context test walks the whole code space in: a
# candidate character before the sigma, one before it behind a cased letter, one after
# the sigma, and one on both sides at once.
PROBE_SHAPES: tuple[Callable[[str], str], ...] = (
    lambda c: c + CAPITAL_SIGMA,
    lambda c: "A" + c + CAPITAL_SIGMA,
    lambda c: "A" + CAPITAL_SIGMA + c,
    lambda c: "A" + c + CAPITAL_SIGMA + c,
)

# Sigma shapes that are not about one code point's properties: words that end in a
# sigma, words that do not, the character that is both cased and case-ignorable, and
# the pair a reviewer found this whole round with.
LITERAL_PROBES: tuple[str, ...] = (
    "\u0391\u03a3",
    "\u0391\u03a3\u0391",
    "\u039f\u0394\u039f\u03a3",
    "\u03a3",
    "\u03a3\u03a3\u03a3",
    "\u0391\u03a3 \u0391\u03a3",
    "\u0391\u0345\u03a3",
    "\u0391\u03a3\u0345",
    "\u0391\u03a3\u00ad",
    "\u0391\u03a3\u00ad\u0391",
    "\u1c8a\u03a3",
    "\u1c8a\u03a3\u1c8a",
    "\ua7cb\u03a3",
    "\u0391\u03a3\ua7cb",
    "\u0391\u03a3\ua7cb\u0391",
    "\u0130\u03a3",
    "\u03a3\u0130",
    "RNA-Seq \u03a3",
    "\U00010400\u03a3",
    "\u03a3\U00010400",
)


def probe_code_points(cased: list[list[int]], case_ignorable: list[list[int]]) -> list[int]:
    """The code points the sampled oracle covers.

    Both edges of every cased and case-ignorable range and the code point just past each
    one, so every place either set starts or stops is in the sample; a stride across the
    whole space for everything in between; and the characters the rule is usually wrong
    about when it is wrong.
    """
    picked: set[int] = set()
    for low, high in [*cased, *case_ignorable]:
        picked.add(low)
        picked.add(high)
        if high + 1 < CODE_SPACE:
            picked.add(high + 1)
    picked.update(range(0, CODE_SPACE, CODE_SPACE // 64))
    picked.update(
        (
            0x41,
            0x61,
            0xAD,
            0x130,
            0x131,
            0x345,
            0x391,
            0x3A3,
            0x3C2,
            0x3C3,
            0x1C8A,
            0xA7CB,
            0x10400,
            0x1D400,
            0xD800,
            0xDFFF,
            0x10FFFF,
        )
    )
    return sorted(picked)


def probe_rows(cased: list[list[int]], case_ignorable: list[list[int]]) -> list[list[str]]:
    """``[the string, what this interpreter lowercases it to]``, for the sample."""
    rows: list[list[str]] = []
    for code_point in probe_code_points(cased, case_ignorable):
        character = chr(code_point)
        for shape in PROBE_SHAPES:
            probe = shape(character)
            rows.append([probe, probe.lower()])
    for probe in LITERAL_PROBES:
        rows.append([probe, probe.lower()])
    return rows


def _render_rows(rows: Iterable[list[object]]) -> str:
    """One row per line, so a table of thousands of entries still reviews as a diff."""
    return "\n".join(f"      {json.dumps(row, separators=(', ', ': '))}," for row in rows)


def _table(name: str, rows: list[list[object]], counted: str, last: bool = False) -> str:
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


def _code_points(rows: list[list[int]]) -> str:
    total = sum(high - low + 1 for low, high in rows)
    return f'    "code_points": {total},\n'


def check_the_rule_is_the_interpreters(tables: Tables) -> None:
    """The tables plus steps 1-4 answer what ``str.lower()`` answers, everywhere.

    Every code point in the space in each of the four probe shapes, plus the literal
    probes: 4,456,448 strings and change. This is the check that lets the other side be
    built from the tables instead of from its own runtime -- if the transcription were
    wrong anywhere, it is wrong here first.
    """
    for code_point in range(CODE_SPACE):
        character = chr(code_point)
        for shape in PROBE_SHAPES:
            probe = shape(character)
            mine = lower_by_the_tables(probe, tables)
            if mine != probe.lower():
                raise AssertionError(
                    f"the written-down rule lowercases {probe!r} to {mine!r} and this "
                    f"interpreter makes it {probe.lower()!r}"
                )
    for probe in LITERAL_PROBES:
        mine = lower_by_the_tables(probe, tables)
        if mine != probe.lower():
            raise AssertionError(
                f"the written-down rule lowercases {probe!r} to {mine!r} and this "
                f"interpreter makes it {probe.lower()!r}"
            )


def check_the_sets_read_the_same_from_both_sides(
    cased: list[list[int]], case_ignorable: list[list[int]]
) -> None:
    """The two sets are the same whichever side of the sigma they are asked from.

    The tables above are derived with the candidate character in FRONT of the sigma,
    which is the rule's step 1 and 2. Steps 3 and 4 read the same two sets from behind
    it, and a set that came out different depending on which loop asked would mean the
    rule is not two sets and two walks at all.
    """
    cased_set = {cp for low, high in cased for cp in range(low, high + 1)}
    ignorable_set = {cp for low, high in case_ignorable for cp in range(low, high + 1)}
    from_behind_cased: set[int] = set()
    from_behind_ignorable: set[int] = set()
    for code_point in range(CODE_SPACE):
        character = chr(code_point)
        # "A\u03a3" + c: the sigma is at index 1 of the lowercased string, because "A"
        # lowercases to one code point. It stops being final exactly when c is cased.
        if ("A" + CAPITAL_SIGMA + character).lower()[1] != FINAL_SIGMA:
            from_behind_cased.add(code_point)
            continue
        # ... and with a cased letter behind c, the sigma stops being final when the
        # forwards walk gets past c, which is what case-ignorable means.
        if ("A" + CAPITAL_SIGMA + character + "A").lower()[1] != FINAL_SIGMA:
            from_behind_ignorable.add(code_point)
    if from_behind_cased != cased_set:
        raise AssertionError(
            "the cased set is not the same asked from behind the sigma: "
            f"{len(from_behind_cased ^ cased_set)} code points differ"
        )
    if from_behind_ignorable != ignorable_set:
        raise AssertionError(
            "the case-ignorable set is not the same asked from behind the sigma: "
            f"{len(from_behind_ignorable ^ ignorable_set)} code points differ"
        )


def build() -> str:
    """The tables file's text, exactly as it is checked in."""
    words = word_char_ranges()
    spaces = whitespace_ranges()
    lowers = lower_mapping()
    cased = cased_ranges()
    case_ignorable = case_ignorable_ranges()

    return (
        "{\n"
        f'  "unicode_version": {json.dumps(PINNED_UNICODE_VERSION)},\n'
        f'  "regenerate": {json.dumps(REGENERATE_COMMAND)},\n'
        '  "note": "Inclusive code point ranges, low first, except lower_map, which is '
        "[code point, the code points str.lower() makes of it]. Read off the interpreter "
        "named above; the TypeScript side is built from this file rather than from its own "
        'tables.",\n'
        + _table("word_chars", list(words), _code_points(words))
        + _table("whitespace", list(spaces), _code_points(spaces))
        + _table("cased", list(cased), _code_points(cased))
        + _table("case_ignorable", list(case_ignorable), _code_points(case_ignorable))
        + _table("lower_map", lowers, "", last=True)
        + "}\n"
    )


def build_probes() -> str:
    """The sampled oracle's text, exactly as it is checked in.

    Strings this interpreter was asked to lowercase and what it answered. The other side
    derives its expectations from the tables -- that is the point of the tables -- so it
    needs somewhere the interpreter's own output is written down verbatim to check the
    derivation against; a sample rather than the whole space, because the whole space in
    four shapes is four and a half million strings and the exhaustive comparison it would
    be worth having lives on this side, in ``check_the_rule_is_the_interpreters``.
    """
    rows = probe_rows(cased_ranges(), case_ignorable_ranges())
    lines = "\n".join(f"    {json.dumps(row, separators=(', ', ': '))}," for row in rows)
    lines = lines[:-1] if lines.endswith(",") else lines
    return (
        "{\n"
        f'  "unicode_version": {json.dumps(PINNED_UNICODE_VERSION)},\n'
        f'  "regenerate": {json.dumps(REGENERATE_COMMAND)},\n'
        '  "note": "[a string, str.lower() of it on the pinned interpreter]. A sample: '
        "both edges of every cased and case-ignorable range and the code point past it, a "
        "stride across the space, the characters this rule is usually wrong about, and a "
        'few whole words.",\n'
        f'  "count": {len(rows)},\n'
        f'  "rows": [\n{lines}\n  ]\n'
        "}\n"
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
    # The two checks the tables have to survive before they are written down. The test
    # module runs them as well, named one each, so a failure says which one broke.
    cased = cased_ranges()
    case_ignorable = case_ignorable_ranges()
    check_the_sets_read_the_same_from_both_sides(cased, case_ignorable)
    check_the_rule_is_the_interpreters(Tables(lower_mapping(), cased, case_ignorable))

    TESTDATA.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(build())
    PROBE_PATH.write_text(build_probes())
    root = Path(__file__).parent.parent
    print(f"wrote {DATA_PATH.relative_to(root)} and {PROBE_PATH.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
