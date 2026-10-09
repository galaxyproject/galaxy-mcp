/**
 * The two copies of the contract interpreter's Unicode tables are the same tables, and what
 * this package does with them is what that interpreter does with them.
 *
 * `src/python-unicode-data.ts` is generated from
 * python/tests/testdata/python-unicode-<version>.json, which that server's suite
 * writes off its own interpreter and keeps current (`uv run python -m tests.python_unicode`,
 * checked by `tests/test_python_unicode.py`). Two checked-in copies of one table can drift, so
 * this reads the original and compares row for row -- a regenerated table on one side and not
 * the other fails here.
 *
 * `pyLower` is checked in three layers, because lowercasing is a mapping and a rule:
 *
 *  1. the mapping, exhaustively: every code point in the space against the interpreter's own
 *     `lower()` as written down in the JSON;
 *  2. the rule, exhaustively: every code point in the space in each of four contexts around a
 *     capital sigma, against a transcription of `handle_capital_sigma` reading the pinned
 *     cased and case-ignorable sets straight out of the JSON. Four and a half million expected
 *     strings are not a file, so this is the layer that is derived rather than recorded -- and
 *     the derivation is not taken on trust either: the other side runs the same transcription
 *     against the live interpreter over the same whole space
 *     (`tests/python_unicode.py::check_the_rule_is_the_interpreters`), so if the rule were
 *     wrong it would be wrong there first;
 *  3. the interpreter's own bytes, sampled: python-lower-probes-<version>.json is 6,392
 *     strings and what `str.lower()` made of each, taken at both edges of every cased and
 *     case-ignorable range, at a stride across the space and at the characters this rule is
 *     usually wrong about. That file ties layers 1 and 2 to output nobody here derived.
 *
 * `isPySpace` is a hand-written literal rather than a generated table, because that set did not
 * differ: the interpreter calls 29 code points whitespace and the literal holds exactly those.
 * It is compared here anyway, code point by code point, so it cannot quietly stop being true.
 */
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, it, expect } from "vitest";
import {
  PY_CASED_RANGES,
  PY_CASE_IGNORABLE_RANGES,
  PY_LOWER_MAP,
  PY_UNICODE_VERSION,
  PY_WORD_CHAR_RANGES,
} from "../src/python-unicode-data";
import { isPyWordChar, isPySpace, pyLower } from "../src/python-str";

const TESTDATA = new URL("../../../../contract/python-unicode/", import.meta.url);

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
  cased: Table;
  case_ignorable: Table;
  lower_map: { count: number; rows: Array<[number, number[]]> };
};

const probes = JSON.parse(
  readFileSync(
    fileURLToPath(new URL(`python-lower-probes-${source.unicode_version}.json`, TESTDATA)),
    "utf8",
  ),
) as { unicode_version: string; count: number; rows: Array<[string, string]> };

const expand = (rows: Array<[number, number]>): Set<number> => {
  const out = new Set<number>();
  for (const [low, high] of rows) for (let code = low; code <= high; code += 1) out.add(code);
  return out;
};

/** The three tables, straight out of the other server's file rather than out of our copy. */
const theirCased = expand(source.cased.rows);
const theirCaseIgnorable = expand(source.case_ignorable.rows);
const theirLower = new Map<number, string>(
  source.lower_map.rows.map(([code, lowered]) => [code, String.fromCodePoint(...lowered)]),
);

/**
 * `str.lower()` written out from those three tables and the rule in the module docstring of
 * tests/python_unicode.py -- a second reading of `handle_capital_sigma`, by index rather than
 * by the slice-and-copy `pyLower` uses, so the two implementations do not share a mistake.
 */
function lowerByTheTables(value: string): string {
  const characters = [...value];
  let out = "";
  for (let i = 0; i < characters.length; i += 1) {
    const character = characters[i]!;
    if (character !== "\u03a3") {
      out += theirLower.get(character.codePointAt(0)!) ?? character;
      continue;
    }
    let behind = i - 1;
    while (behind >= 0 && theirCaseIgnorable.has(characters[behind]!.codePointAt(0)!)) behind -= 1;
    let final = behind >= 0 && theirCased.has(characters[behind]!.codePointAt(0)!);
    if (final) {
      let ahead = i + 1;
      while (ahead < characters.length && theirCaseIgnorable.has(characters[ahead]!.codePointAt(0)!))
        ahead += 1;
      final = ahead === characters.length || !theirCased.has(characters[ahead]!.codePointAt(0)!);
    }
    out += final ? "\u03c2" : "\u03c3";
  }
  return out;
}

describe("the pinned Unicode tables", () => {
  it("are one interpreter's, so there is one file to copy from", () => {
    expect(names).toEqual([`python-unicode-${source.unicode_version}.json`]);
    expect(probes.unicode_version).toBe(source.unicode_version);
  });

  it("are the version this package says it follows", () => {
    expect(PY_UNICODE_VERSION).toBe(source.unicode_version);
  });

  it("hold the same word-character ranges here as there, range for range", () => {
    expect(PY_WORD_CHAR_RANGES.map(([low, high]) => [low, high])).toEqual(source.word_chars.rows);
    expect(PY_WORD_CHAR_RANGES.length).toBe(source.word_chars.count);
  });

  it("hold the same cased and case-ignorable sets here as there, range for range", () => {
    expect(PY_CASED_RANGES.map(([low, high]) => [low, high])).toEqual(source.cased.rows);
    expect(PY_CASED_RANGES.length).toBe(source.cased.count);
    expect(PY_CASE_IGNORABLE_RANGES.map(([low, high]) => [low, high])).toEqual(
      source.case_ignorable.rows,
    );
    expect(PY_CASE_IGNORABLE_RANGES.length).toBe(source.case_ignorable.count);
    // The two sets do not overlap, because a character that is both is walked past rather
    // than counted as cased -- which is the precedence the interpreter's loops have.
    for (const [low, high] of PY_CASED_RANGES) {
      for (let code = low; code <= high; code += 1) expect(theirCaseIgnorable.has(code)).toBe(false);
    }
  });

  it("hold the same lowercase mapping here as there, entry for entry", () => {
    expect(PY_LOWER_MAP.map(([code, lowered]) => [code, lowered])).toEqual(
      source.lower_map.rows.map(([code, lowered]) => [code, String.fromCodePoint(...lowered)]),
    );
    expect(PY_LOWER_MAP.length).toBe(source.lower_map.count);
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
    const whitespace = expand(source.whitespace.rows);
    const mine = new Set<number>();
    for (let code = 0; code < 0x110000; code += 1) {
      if (isPySpace(code)) mine.add(code);
    }
    expect([...mine].sort((a, b) => a - b)).toEqual([...whitespace].sort((a, b) => a - b));
  });
});

describe("pyLower is the interpreter's lower()", () => {
  it("for every code point in the space, on its own", () => {
    const wrong: string[] = [];
    for (let code = 0; code < 0x110000; code += 1) {
      const character = String.fromCodePoint(code);
      const expected = theirLower.get(code) ?? character;
      if (pyLower(character) !== expected) wrong.push(code.toString(16));
    }
    expect(wrong).toEqual([]);
  });

  it("for every code point in the space, in four contexts around a capital sigma", () => {
    const wrong: string[] = [];
    for (let code = 0; code < 0x110000; code += 1) {
      const c = String.fromCodePoint(code);
      for (const probe of [c + "\u03a3", "A" + c + "\u03a3", "A\u03a3" + c, "A" + c + "\u03a3" + c]) {
        if (pyLower(probe) !== lowerByTheTables(probe)) {
          wrong.push(`${code.toString(16)}: ${JSON.stringify(probe)}`);
          break;
        }
      }
    }
    expect(wrong).toEqual([]);
  });

  it("on the strings that interpreter was actually asked to lowercase", () => {
    const wrong: string[] = [];
    for (const [input, expected] of probes.rows) {
      if (pyLower(input) !== expected) {
        wrong.push(`${JSON.stringify(input)} -> ${JSON.stringify(pyLower(input))}`);
      }
    }
    expect(wrong).toEqual([]);
    expect(probes.rows.length).toBe(probes.count);
    // The sample is worth having only if it is big enough to be a sample.
    expect(probes.count).toBeGreaterThan(5000);
  });

  it("without asking this runtime anything about Unicode", () => {
    /**
     * The point of the round, as a source check: a lowercasing that hands any part of the
     * string to `toLowerCase` is reading Node's Unicode edition for that part, whatever the
     * tables say about the rest -- which is how a contextual rule slipped back in after the
     * mapping had been pinned. So the functions that make up `pyLower` may not mention it,
     * nor a Unicode property class, nor anything else that resolves against this runtime's
     * tables.
     */
    const text = readFileSync(fileURLToPath(new URL("../src/python-str.ts", import.meta.url)), "utf8");
    const body = (name: string): string => {
      const at = text.indexOf(`function ${name}(`);
      expect(at).toBeGreaterThan(-1);
      const end = text.indexOf("\n}\n", at);
      return text.slice(at, end);
    };
    for (const name of ["pyLower", "isFinalSigma", "isPyCased", "isPyCaseIgnorable"]) {
      const source = body(name);
      for (const forbidden of ["toLowerCase", "toUpperCase", "\\p{", "localeCompare", "normalize("]) {
        expect([name, source.includes(forbidden)]).toEqual([name, false]);
      }
    }
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

describe("the case tables are the data too", () => {
  it("leave a character this runtime cases and the pinned edition does not uncased", () => {
    // U+1C8A got a case after 15.0.0. To this runtime it is a cased letter, so a sigma after
    // it is at the end of a word; to the contract interpreter it is nothing at all.
    expect(/\p{Cased}/u.test("\u1c8a")).toBe(true);
    expect(expand(source.cased.rows).has(0x1c8a)).toBe(false);
    expect(pyLower("\u1c8a\u03a3")).toBe("\u1c8a\u03c3");
    expect("\u1c8a\u03a3".toLowerCase()).toBe("\u1c8a\u03c2");
  });

  it("walk past a character that is both cased and case-ignorable", () => {
    // U+0345 is cased by property and case-ignorable by property, and the interpreter asks
    // case-ignorable first -- so it is not in the cased table and a sigma behind it looks
    // past it for a cased letter.
    expect(theirCaseIgnorable.has(0x345)).toBe(true);
    expect(theirCased.has(0x345)).toBe(false);
    // That interpreter makes "\u0391\u0345\u03a3" into "\u03b1\u0345\u03c2": the walk back skips the
    // ypogegrammeni, finds the alpha, and the ypogegrammeni lowercases to itself.
    expect(pyLower("\u0391\u0345\u03a3")).toBe("\u03b1\u0345\u03c2");
  });

  it("walk past a soft hyphen, which is the everyday case-ignorable", () => {
    // "\u0391\u03a3\u00ad".lower() is "\u03b1\u03c2\u00ad" and "\u0391\u03a3\u00ad\u0391".lower() is
    // "\u03b1\u03c3\u00ad\u03b1" -- the hyphen is skipped and the alpha behind it is not.
    expect(pyLower("\u0391\u03a3\u00ad")).toBe("\u03b1\u03c2\u00ad");
    expect(pyLower("\u0391\u03a3\u00ad\u0391")).toBe("\u03b1\u03c3\u00ad\u03b1");
  });
});
