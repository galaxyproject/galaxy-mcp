"""The checked-in Unicode tables still say what this interpreter says.

``tests/testdata/python-unicode-15.0.0.json`` is the contract the TypeScript
tokeniser and text helpers are built from: they do not ask Node's tables anything,
because Node's tables are a different edition of Unicode and would answer differently.
A file that has stopped matching the interpreter is worse than no file, so this
regenerates it and compares.

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
    REGENERATE_COMMAND,
    build,
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
        f"{DATA_PATH.name} is missing; run `{REGENERATE_COMMAND}` from mcp-server-galaxy-py/"
    )


def test_the_tables_still_match_the_interpreter() -> None:
    assert DATA_PATH.read_text() == build(), (
        f"{DATA_PATH.name} no longer says what this interpreter says; run "
        f"`{REGENERATE_COMMAND}` from mcp-server-galaxy-py/ and review the diff -- the "
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
