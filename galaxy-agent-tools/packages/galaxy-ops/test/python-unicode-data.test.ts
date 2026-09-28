/**
 * The two copies of the contract interpreter's Unicode tables are the same tables.
 *
 * `src/python-unicode-data.ts` is generated from
 * mcp-server-galaxy-py/tests/testdata/python-unicode-<version>.json, which that server's suite
 * writes off its own interpreter and keeps current (`uv run python -m tests.python_unicode`,
 * checked by `tests/test_python_unicode.py`). Two checked-in copies of one table can drift, so
 * this reads the original and compares row for row -- a regenerated table on one side and not
 * the other fails here.
 *
 * `isPySpace` is a hand-written literal rather than a generated table, because that set did not
 * differ: the interpreter calls 29 code points whitespace and the literal holds exactly those.
 * It is compared here anyway, code point by code point, so it cannot quietly stop being true.
 */
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, it, expect } from "vitest";
import {
  PY_LOWER_EXCEPTION_RANGES,
  PY_UNICODE_VERSION,
  PY_WORD_CHAR_RANGES,
} from "../src/python-unicode-data";
import { isPyWordChar, isPySpace, pyLower } from "../src/python-str";

const TESTDATA = new URL("../../../../mcp-server-galaxy-py/tests/testdata/", import.meta.url);

interface Table {
  count: number;
  code_points?: number;
  rows: Array<[number, number]>;
}

const names = readdirSync(fileURLToPath(TESTDATA)).filter(
  (name) => name.startsWith("python-unicode-") && name.endsWith(".json"),
);

const source = JSON.parse(readFileSync(fileURLToPath(new URL(names[0]!, TESTDATA)), "utf8")) as {
  unicode_version: string;
  word_chars: Table;
  whitespace: Table;
  lower_map: { count: number; rows: Array<[number, string]> };
};

describe("the pinned Unicode tables", () => {
  it("are one interpreter's, so there is one file to copy from", () => {
    expect(names).toEqual([`python-unicode-${source.unicode_version}.json`]);
  });

  it("are the version this package says it follows", () => {
    expect(PY_UNICODE_VERSION).toBe(source.unicode_version);
  });

  it("hold the same word-character ranges here as there, range for range", () => {
    expect(PY_WORD_CHAR_RANGES.map(([low, high]) => [low, high])).toEqual(source.word_chars.rows);
    expect(PY_WORD_CHAR_RANGES.length).toBe(source.word_chars.count);
  });

  it("are read by isPyWordChar exactly as written, at every edge", () => {
    for (const [low, high] of source.word_chars.rows) {
      expect(isPyWordChar(low)).toBe(true);
      expect(isPyWordChar(high)).toBe(true);
      if (low > 0) expect(isPyWordChar(low - 1)).toBe(false);
      if (high < 0x10ffff) expect(isPyWordChar(high + 1)).toBe(false);
    }
  });

  it("agree with isPySpace about whitespace, code point by code point", () => {
    const whitespace = new Set<number>();
    for (const [low, high] of source.whitespace.rows) {
      for (let code = low; code <= high; code += 1) whitespace.add(code);
    }
    const mine = new Set<number>();
    for (let code = 0; code < 0x110000; code += 1) {
      if (isPySpace(code)) mine.add(code);
    }
    expect([...mine].sort((a, b) => a - b)).toEqual([...whitespace].sort((a, b) => a - b));
  });
});

describe("the lowercasing exceptions", () => {
  /**
   * The list of code points to leave alone is worked out by comparing the two runtimes, so
   * unlike the word table it is not complete by construction: a runtime whose Unicode data
   * moves on could lowercase a 56th. This recomputes it over the whole code space against the
   * interpreter's own mapping, so that arrives as a failure here with the command to
   * regenerate, rather than as a search quietly matching something the other server does not.
   */
  it("are every code point this runtime and the interpreter case differently", () => {
    const theirs = new Map(source.lower_map.rows);
    const disagree: number[] = [];
    for (let code = 0; code < 0x110000; code += 1) {
      const character = String.fromCodePoint(code);
      if ((theirs.get(code) ?? character) !== character.toLowerCase()) disagree.push(code);
    }
    const listed: number[] = [];
    for (const [low, high] of PY_LOWER_EXCEPTION_RANGES) {
      for (let code = low; code <= high; code += 1) listed.push(code);
    }
    expect(listed).toEqual(disagree);
  });

  it("make pyLower the interpreter's lower() for every code point in the space", () => {
    const theirs = new Map(source.lower_map.rows);
    const wrong: string[] = [];
    for (let code = 0; code < 0x110000; code += 1) {
      const character = String.fromCodePoint(code);
      const expected = theirs.get(code) ?? character;
      if (pyLower(character) !== expected) wrong.push(code.toString(16));
    }
    expect(wrong).toEqual([]);
  });
});

describe("the word table is the data, not this runtime's opinion of it", () => {
  it("calls a letter assigned after the pinned edition a separator", () => {
    // U+A7CB is a letter to this runtime's `\p{L}` and unassigned to the pinned tables, which
    // is the whole regression: a boundary lands beside it there and not here.
    expect(/\p{L}/u.test("Ɤ")).toBe(true);
    expect(isPyWordChar(0xa7cb)).toBe(false);
  });

  it("calls a letter assigned before it a word character", () => {
    expect(isPyWordChar(0xa7ca)).toBe(true);
  });

  it("counts a digit and an underscore in, as `\\w` does and `\\p{L}` does not", () => {
    expect(isPyWordChar(0x30)).toBe(true);
    expect(isPyWordChar(0x5f)).toBe(true);
    expect(isPyWordChar(0x20)).toBe(false);
  });

  it("counts an astral letter in and a lone surrogate out", () => {
    expect(isPyWordChar(0x1d41a)).toBe(true);
    expect(isPyWordChar(0xd800)).toBe(false);
  });
});
