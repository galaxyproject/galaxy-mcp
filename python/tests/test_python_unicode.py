"""The checked-in Unicode tables still say what this interpreter says.

``tests/testdata/python-unicode-15.0.0.json`` is the contract the TypeScript
tokeniser and text helpers are built from: they do not ask Node's tables anything,
because Node's tables are a different edition of Unicode and would answer differently.
A file that has stopped matching the interpreter is worse than no file, so this
regenerates it and compares.

The tables carry the final-sigma rule's two sets as well as the one-code-point mapping,
and the rule itself is transcribed in ``tests/python_unicode.py``. Two of the checks
below are about that transcription rather than about the file: that the two sets come
out the same asked from either side of the sigma, and that the rule plus the tables
lowercase every code point in the space, in four contexts each, exactly the way this
interpreter does. That is what lets the other runtime lowercase out of the tables
instead of out of its own Unicode edition.

CI runs this on 3.10 and 3.11 as well, which carry Unicode 13.0 and 14.0. There the
tables would be a different set and the comparison would mean nothing, so the checks
skip with the version they found -- the pinned tables belong to one interpreter, and
saying so is the point of the file's name.
"""

from __future__ import annotations

import json
import re
import unicodedata

import pytest

from .python_unicode import (
    DATA_PATH,
    PINNED_UNICODE_VERSION,
    PROBE_PATH,
    REGENERATE_COMMAND,
    Tables,
    build,
    build_probes,
    case_ignorable_ranges,
    cased_ranges,
    check_the_rule_is_the_interpreters,
    check_the_sets_read_the_same_from_both_sides,
    lower_mapping,
)

pytestmark = pytest.mark.skipif(
    unicodedata.unidata_version != PINNED_UNICODE_VERSION,
    reason=(
        f"the pinned Unicode tables are {PINNED_UNICODE_VERSION} and this interpreter "
        f"carries {unicodedata.unidata_version}; only the pinned interpreter can say "
        "whether the checked-in file is current"
    ),
)


def test_the_file_is_there() -> None:
    assert DATA_PATH.exists(), (
        f"{DATA_PATH.name} is missing; run `{REGENERATE_COMMAND}` from python/"
    )
    assert PROBE_PATH.exists(), (
        f"{PROBE_PATH.name} is missing; run `{REGENERATE_COMMAND}` from python/"
    )


def test_the_tables_still_match_the_interpreter() -> None:
    assert DATA_PATH.read_text() == build(), (
        f"{DATA_PATH.name} no longer says what this interpreter says; run "
        f"`{REGENERATE_COMMAND}` from python/ and review the diff -- the "
        "TypeScript tokeniser and text helpers are built from this file, so a change here "
        "changes what both servers answer"
    )


def test_the_file_names_the_version_it_holds() -> None:
    written = json.loads(DATA_PATH.read_text())
    assert written["unicode_version"] == unicodedata.unidata_version
    assert DATA_PATH.name == f"python-unicode-{written['unicode_version']}.json"


def test_the_word_table_is_the_set_a_boundary_reads() -> None:
    """A spot check that the table is about `\\b` and not just about `\\w`.

    U+A7CB is a letter Unicode added after 15.0, so this interpreter reads it as a
    separator: `rnaseq` beside it is a word, and a tokeniser on newer tables sees one
    long word and no searchable term at all. That difference is a golden fixture
    (`recommend_iwc_workflows/letter_added_after_unicode_15`), and this is the fact it
    rests on.
    """
    written = json.loads(DATA_PATH.read_text())
    rows = written["word_chars"]["rows"]
    assert not any(low <= 0xA7CB <= high for low, high in rows)
    assert re.findall(r"\b[a-zA-Z]{2,}\b", "rnaseqꟋ") == ["rnaseq"]
    # The letter before it in the same block has been assigned since 5.1, so it is a
    # word character and there is no boundary beside it.
    assert any(low <= 0xA7CA <= high for low, high in rows)
    assert re.findall(r"\b[a-zA-Z]{2,}\b", "rnaseqꟊ") == []


def test_the_probe_file_is_current() -> None:
    assert PROBE_PATH.read_text() == build_probes(), (
        f"{PROBE_PATH.name} no longer says what this interpreter says; run "
        f"`{REGENERATE_COMMAND}` from python/ and review the diff"
    )


def test_every_probe_is_this_interpreters_own_answer() -> None:
    """The oracle is output, not expectation: each row is a string and its `lower()`."""
    written = json.loads(PROBE_PATH.read_text())
    assert written["count"] == len(written["rows"])
    for probe, expected in written["rows"]:
        assert probe.lower() == expected


def test_the_case_sets_read_the_same_from_either_side_of_the_sigma() -> None:
    """Step 1-2 of the rule and step 3 have to be asking about the same two sets."""
    check_the_sets_read_the_same_from_both_sides(cased_ranges(), case_ignorable_ranges())


def test_the_tables_and_the_rule_are_this_interpreters_lower() -> None:
    """Every code point in the space, in each of the four contexts the other side tests.

    The whole point of writing the sets down: a transcription of `handle_capital_sigma`
    over these tables answers what `str.lower()` answers, so the other runtime can apply
    it without asking its own Unicode edition anything.
    """
    check_the_rule_is_the_interpreters(
        Tables(lower_mapping(), cased_ranges(), case_ignorable_ranges())
    )


def test_the_sigma_after_a_letter_assigned_after_the_pinned_edition() -> None:
    """The case this round is about, as one line of interpreter output.

    U+1C8A was given a case after 15.0.0, so this interpreter reads it as uncased: the
    sigma after it is not at the end of a word and lowercases to an ordinary sigma. A
    runtime on newer tables makes it a final sigma, and a search for one does not find
    the other -- which is the golden fixture `search_tools_by_name/sigma_after_uncased`.
    """
    assert "\u1c8a\u03a3".lower() == "\u1c8a\u03c3"
    assert "\u0391\u03a3".lower() == "\u03b1\u03c2"


def test_a_lone_surrogate_is_not_half_of_an_astral_character() -> None:
    """`in` is a question about code points, which is not what the other runtime asks.

    An astral character is one code point here and two sixteen-bit units there, so
    `String.prototype.includes` reports the low half of an emoji as a substring of that
    emoji and this reports nothing. A client can put a lone surrogate in a query -- JSON
    carries one as a `\\udE00` escape and both parsers hand it back as a character -- so
    the TypeScript search tools go through `pyContains` rather than `includes`.
    """
    assert "\ude00" not in "\U0001f600"
    assert "\ud83d" not in "\U0001f600"
    assert "\ude00" in "a\ude00b"
