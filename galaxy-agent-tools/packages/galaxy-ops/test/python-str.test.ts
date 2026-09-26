/**
 * The Python string semantics the recommender port leans on, held to the interpreter.
 *
 * Each case below is one CPython answers a particular way and JavaScript's look-alike answers
 * the other way, so a regression shows up here rather than as two surfaces recommending two
 * different containers. The expected values were read off the Galaxy checkout's interpreter
 * (Python 3.12, `str.strip`, `str.isspace`, `sorted`) rather than recalled.
 */
import { describe, it, expect } from "vitest";
import { GalaxyValidationError } from "../src/errors";
import {
  comparePyStrings,
  isPySpace,
  pyIntFromDigits,
  pyRepr,
  pyReprList,
  pyStrip,
  pyUtf8EncodeError,
  pyZfill,
} from "../src/python-str";

// ---------------------------------------------------------------------------
// pyStrip -- str.strip(), which is not String.prototype.trim()
// ---------------------------------------------------------------------------

describe("pyStrip", () => {
  it.each([
    ["\t 1.17 \n", "1.17"],
    ["\r\x0b\x0c1.17", "1.17"],
    ["1.17", "1.17"],
    ["1.17", "1.17"],
    [" 1.17 ", "1.17"],
    [" 1.17 ", "1.17"],
    ["　1.17 ", "1.17"],
    ["1.17", "1.17"],
    ["", ""],
    ["   ", ""],
    // Kept, because Python's str.isspace() is false for it.
    ["﻿1.17﻿", "﻿1.17﻿"],
    // And a kept character stops the scan, so the space behind it stays too.
    ["﻿ 1.17", "﻿ 1.17"],
    // Inner whitespace is never touched, which is what leaves it for build_target to refuse.
    ["1.17 2", "1.17 2"],
  ])("strips %j down to %j", (raw, stripped) => {
    expect(pyStrip(raw)).toBe(stripped);
  });

  it("differs from JavaScript's trim on exactly five code points", () => {
    // Python's set is bidi class WS/B/S plus category Zs; JavaScript's is Unicode White_Space
    // without U+0085 and with U+FEFF. Both directions change what this tool resolves.
    for (const c of ["", "", "", "", ""]) {
      expect(pyStrip(`${c}x${c}`)).toBe("x");
      expect(`${c}x${c}`.trim()).toBe(`${c}x${c}`);
    }
    expect(pyStrip("﻿x﻿")).toBe("﻿x﻿");
    expect("﻿x﻿".trim()).toBe("x");
  });

  it("covers the 29 code points str.isspace() is true for and no others", () => {
    // The interpreter's own answer, code point by code point over the whole space.
    const expected = [
      0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680, 0x2000,
      0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028,
      0x2029, 0x202f, 0x205f, 0x3000,
    ];
    const actual: number[] = [];
    for (let c = 0; c <= 0x10ffff; c += 1) if (isPySpace(c)) actual.push(c);
    expect(actual).toEqual(expected);
  });
});

// ---------------------------------------------------------------------------
// comparePyStrings -- Python's `<` on str, which is not JavaScript's on string
// ---------------------------------------------------------------------------

describe("comparePyStrings", () => {
  it("orders by code point where JavaScript orders by UTF-16 code unit", () => {
    const astral = "\u{10000}";
    const pua = "";
    // Python: '' < '\U00010000'. JavaScript: the surrogate pair starts at U+D800.
    expect(comparePyStrings(pua, astral)).toBe(-1);
    expect(comparePyStrings(astral, pua)).toBe(1);
    expect(pua < astral).toBe(false);
  });

  it("agrees with JavaScript everywhere below the surrogate range", () => {
    const words = ["", "a", "ab", "b", "0", "9", "*final", "*final-", "", " ", "퟿"];
    for (const a of words) {
      for (const b of words) {
        const js = a < b ? -1 : a > b ? 1 : 0;
        expect(comparePyStrings(a, b)).toBe(js);
      }
    }
  });

  it("falls back to the shorter string, as a Python comparison does", () => {
    expect(comparePyStrings("ab", "abc")).toBe(-1);
    expect(comparePyStrings("abc", "ab")).toBe(1);
    expect(comparePyStrings("abc", "abc")).toBe(0);
    expect(comparePyStrings("", "")).toBe(0);
    expect(comparePyStrings("", "a")).toBe(-1);
  });

  it("compares a lone surrogate as its own code point, as Python does", () => {
    // json.loads('"\\ud800"') gives Python a string holding U+D800, and it orders on that
    // value; iterating a JavaScript string yields the unpaired surrogate the same way.
    expect(comparePyStrings("\ud800", "")).toBe(-1);
    expect(comparePyStrings("\ud800", "\u{10000}")).toBe(-1);
  });
});

// ---------------------------------------------------------------------------
// pyZfill -- str.zfill(), which counts code points where padStart counts units
// ---------------------------------------------------------------------------

describe("pyZfill", () => {
  it.each([
    // A legacy version part carrying MATHEMATICAL DOUBLE-STRUCK DIGIT ZERO, which is two
    // UTF-16 units and one code point.
    ["1\u{1D7D8}", 8, "0000001\u{1D7D8}"],
    ["12", 8, "00000012"],
    ["123456789", 8, "123456789"],
    ["\u{1D7D8}12", 8, "00000\u{1D7D8}12"],
    // zfill puts the fill after a sign; nothing in the recommender reaches this.
    ["-1", 4, "-001"],
    ["+2", 4, "+002"],
  ])("zfills %j to %i as Python does", (value, width, expected) => {
    expect(pyZfill(value, width)).toBe(expected);
  });

  it("is not padStart, which comes up a code point short", () => {
    expect("1\u{1D7D8}".padStart(8, "0")).toBe("000001\u{1D7D8}");
    expect(pyZfill("1\u{1D7D8}", 8)).not.toBe("1\u{1D7D8}".padStart(8, "0"));
  });

  it("counts a lone surrogate as one code point, as len() does", () => {
    expect(pyZfill("\ud800", 3)).toBe("00\ud800");
  });
});

// ---------------------------------------------------------------------------
// pyIntFromDigits -- int() over a \d+ match, which is Nd and not [0-9]
// ---------------------------------------------------------------------------

describe("pyIntFromDigits", () => {
  it.each([
    ["2", 2n],
    ["007", 7n],
    // ARABIC-INDIC DIGIT TWO, which BigInt() throws on and int() reads as 2.
    ["٢", 2n],
    ["١٠", 10n],
    // FULLWIDTH DIGIT SEVEN, and two astral digits from the mathematical sets.
    ["７", 7n],
    ["\u{1D7D9}\u{1D7D8}", 10n],
    // Past 2^53, where a JS number stops being able to tell two versions apart.
    ["9007199254740993", 9007199254740993n],
  ])("reads %j as %s", (digits, expected) => {
    expect(pyIntFromDigits(digits)).toBe(expected);
  });

  it("refuses what int() refuses, rather than returning a quiet zero", () => {
    expect(() => pyIntFromDigits("")).toThrow();
    expect(() => pyIntFromDigits("1a")).toThrow();
    // The superscripts are No, not Nd: `\d` does not match them and int() will not read them.
    expect(() => pyIntFromDigits("¹")).toThrow();
  });

  it("converts 4300 digits and refuses 4301, which is int()'s boundary", () => {
    // sys.get_int_max_str_digits() is 4300 by default from CPython 3.11 on, and int() raises
    // rather than converting past it. BigInt has no such limit, so a tag whose build number is
    // 4301 nines used to be a number here and a ValueError out of parse_tag there.
    expect(pyIntFromDigits("9".repeat(4300))).toBe(BigInt("9".repeat(4300)));
    expect(() => pyIntFromDigits("9".repeat(4301))).toThrow(GalaxyValidationError);
    expect(() => pyIntFromDigits("9".repeat(4301))).toThrow(
      "Exceeds the limit (4300 digits) for integer string conversion: value has 4301 digits; " +
        "use sys.set_int_max_str_digits() to increase the limit",
    );
  });

  it("counts the digits as written, so leading zeros count", () => {
    // int("0" * 4300 + "9") is refused too, and reports 4301 digits -- the limit is on the
    // string, not on the value it would have produced.
    expect(() => pyIntFromDigits("0".repeat(4300) + "9")).toThrow("value has 4301 digits");
    expect(pyIntFromDigits("0".repeat(4299) + "9")).toBe(9n);
  });

  it("counts an astral digit once, as Python counts code points", () => {
    // MATHEMATICAL MONOSPACE DIGIT NINE is two UTF-16 units and one digit; counting units would
    // refuse 2151 of them, where int() takes 4300.
    const nine = "\u{1D7FF}";
    expect(pyIntFromDigits(nine.repeat(4300))).toBe(BigInt("9".repeat(4300)));
    expect(() => pyIntFromDigits(nine.repeat(4301))).toThrow("value has 4301 digits");
  });
});

// ---------------------------------------------------------------------------
// pyRepr -- repr(), which an f-string uses for every element of a container
// ---------------------------------------------------------------------------

describe("pyRepr", () => {
  it.each([
    // Every expected value below is what `python -B` printed for `repr(...)`.
    ["a\\b", "'a\\\\b'"],
    ["c", "'c'"],
    ["samtools", "'samtools'"],
    // A single quote in the string moves the delimiter to a double one ...
    ["it's", '"it\'s"'],
    // ... unless a double quote is in there too, in which case the single one is escaped.
    ['both \' and "', "'both \\' and \"'"],
    ['say "hi"', "'say \"hi\"'"],
    // \x for the C0 controls and U+007F, and U+0085 is a control too.
    ["a\u0000b\u007f\u0085", "'a\\x00b\\x7f\\x85'"],
    ["\t\n\r", "'\\t\\n\\r'"],
    // Cf, Zl and an astral Cf: non-printable, so escaped by width.
    ["\u200b\u2028\u{E0001}", "'\\u200b\\u2028\\U000e0001'"],
    // Zs other than U+0020, which is the one space repr leaves alone.
    ["\u00a0 ", "'\\xa0 '"],
    // A lone surrogate is Cs, so it escapes rather than being substituted.
    ["a\ud800", "'a\\ud800'"],
    // Printable astral characters stay themselves.
    ["\u{1F600}\u{10000}", "'\u{1F600}\u{10000}'"],
    ["", "''"],
  ])("reprs %j as Python does", (value, expected) => {
    expect(pyRepr(value)).toBe(expected);
  });
});

describe("pyReprList", () => {
  it.each([
    [["a\\b", "c"], "['a\\\\b', 'c']"],
    [["samtools", "bwa"], "['samtools', 'bwa']"],
    [["it's", 'say "hi"'], "[\"it's\", 'say \"hi\"']"],
    [[], "[]"],
    [["one"], "['one']"],
  ])("reprs %j as Python does", (values, expected) => {
    expect(pyReprList(values)).toBe(expected);
  });
});

// ---------------------------------------------------------------------------
// pyUtf8EncodeError -- str.encode(), which refuses a surrogate
// ---------------------------------------------------------------------------

describe("pyUtf8EncodeError", () => {
  it("says nothing about a string Python can encode", () => {
    expect(pyUtf8EncodeError("samtools")).toBeNull();
    // A well-formed pair is one astral code point, which encodes fine.
    expect(pyUtf8EncodeError("a\u{10000}b")).toBeNull();
    expect(pyUtf8EncodeError("")).toBeNull();
  });

  it("gives CPython's message for one unencodable code point", () => {
    expect(pyUtf8EncodeError("a\ud800\nb")).toBe(
      "'utf-8' codec can't encode character '\\ud800' in position 1: surrogates not allowed",
    );
    expect(pyUtf8EncodeError("https://quay.io/api/v1/repository/biocontainers/a\ud800")).toBe(
      "'utf-8' codec can't encode character '\\ud800' in position 49: surrogates not allowed",
    );
  });

  it("reports a run of them as a range, as CPython's encoder does", () => {
    expect(pyUtf8EncodeError("x\udfff\udfff")).toBe(
      "'utf-8' codec can't encode characters in position 1-2: surrogates not allowed",
    );
  });
});
