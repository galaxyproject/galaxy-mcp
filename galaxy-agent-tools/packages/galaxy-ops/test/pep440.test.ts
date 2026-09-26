/**
 * The version ordering `version_sorted` runs on, held to `packaging`'s own rules.
 *
 * Galaxy sorts biocontainer tags by `galaxy.tool_util.version.parse_version`, which is
 * `packaging.version.Version` with packaging 21.3's `LegacyVersion` as the fallback. If this
 * port disagrees anywhere, the two surfaces pick a different "newest" tag for the same
 * repository -- which a hand-rolled comparator did, on `1.17` vs `1.17.1`, before this
 * existed. So the tables below are the orderings packaging's rules produce, written out and
 * asserted rather than described in a comment.
 *
 * The first table is PEP 440's own "Summary of permitted suffixes and relative ordering"
 * example, in its published order; the rest are the individual rules that example does not
 * reach (release normalisation, epochs, the spelling aliases, the legacy fallback).
 */
import { describe, it, expect } from "vitest";
import { compareVersions, parseVersion, compareVersionKeys } from "../src/pep440";
import { GalaxyValidationError } from "../src/errors";

/** Assert `ordered` is strictly ascending, pair by pair and as a whole sort. */
function expectAscending(ordered: readonly string[]): void {
  for (let i = 1; i < ordered.length; i++) {
    const [lo, hi] = [ordered[i - 1]!, ordered[i]!];
    expect(compareVersions(lo, hi), `${lo} < ${hi}`).toBeLessThan(0);
    expect(compareVersions(hi, lo), `${hi} > ${lo}`).toBeGreaterThan(0);
  }
  // Sorting a rotation has to land back on the published order, which no amount of
  // pairwise agreement guarantees on its own if the comparator is not transitive.
  const rotated = [...ordered.slice(7), ...ordered.slice(0, 7)];
  expect([...rotated].sort(compareVersions)).toEqual([...ordered]);
}

/** Assert every member of `equal` compares equal to every other. */
function expectAllEqual(equal: readonly string[]): void {
  for (const a of equal) {
    for (const b of equal) {
      expect(compareVersions(a, b), `${a} == ${b}`).toBe(0);
    }
  }
}

describe("compareVersions -- PEP 440's published ordering", () => {
  // PEP 440, "Summary of permitted suffixes and relative ordering", verbatim.
  const canonical = [
    "1.0.dev456",
    "1.0a1",
    "1.0a2.dev456",
    "1.0a12.dev456",
    "1.0a12",
    "1.0b1.dev456",
    "1.0b2",
    "1.0b2.post345.dev456",
    "1.0b2.post345",
    "1.0rc1.dev456",
    "1.0rc1",
    "1.0",
    "1.0+abc.5",
    "1.0+abc.7",
    "1.0+5",
    "1.0.post456.dev34",
    "1.0.post456",
    "1.0.1",
    "1.1.dev1",
  ];

  it("orders the whole example exactly as PEP 440 lists it", () => {
    expectAscending(canonical);
  });
});

describe("compareVersions -- the release segment", () => {
  it("compares release numbers numerically, not as text", () => {
    // The case that broke the hand-rolled comparator: 1.17 is not newer than 1.17.1.
    expectAscending(["1.9", "1.10", "1.16", "1.17", "1.17.1", "1.18", "2.0", "10.0"]);
  });

  it("ignores trailing zeros, so 1.0 and 1.0.0 are the same version", () => {
    expectAllEqual(["1", "1.0", "1.0.0", "1.0.0.0"]);
    expectAllEqual(["2", "2.0.0"]);
  });

  it("drops a leading v and surrounding whitespace", () => {
    expectAllEqual(["1.0", "v1.0", " 1.0 ", "\tv1.0\n"]);
  });
});

describe("compareVersions -- epochs", () => {
  it("lets an epoch outrank any release", () => {
    expectAscending(["10.0", "1!0.1", "1!1.0", "2!0.1"]);
  });

  it("treats a 0 epoch as no epoch", () => {
    expectAllEqual(["1.0", "0!1.0"]);
  });
});

describe("compareVersions -- pre, post and dev", () => {
  it("normalises the pre-release spellings packaging accepts", () => {
    expectAllEqual(["1.0a1", "1.0alpha1", "1.0.a.1", "1.0-a-1", "1.0_alpha_1"]);
    expectAllEqual(["1.0b2", "1.0beta2"]);
    expectAllEqual(["1.0rc1", "1.0c1", "1.0.pre1", "1.0.preview1"]);
  });

  it("normalises the post-release spellings, including the bare -N form", () => {
    expectAllEqual(["1.0.post1", "1.0-1", "1.0.rev1", "1.0.r1", "1.0_post_1"]);
  });

  it("reads a missing numeral as 0", () => {
    expectAllEqual(["1.0a", "1.0a0"]);
    expectAllEqual(["1.0.post", "1.0.post0"]);
    expectAllEqual(["1.0.dev", "1.0.dev0"]);
  });

  it("puts a post-release above the release and a dev release below it", () => {
    expectAscending(["1.0.dev1", "1.0", "1.0.post1", "1.0.post2", "1.0.1"]);
  });

  it("puts a plain dev release below every pre-release of the same version", () => {
    // `pre` is NegativeInfinity when there is a dev but no pre and no post, which is the
    // only reason 1.0.dev0 sorts under 1.0a0 rather than over it.
    expectAscending(["1.0.dev0", "1.0a0.dev0", "1.0a0", "1.0b0", "1.0rc0", "1.0"]);
  });
});

describe("compareVersions -- local versions", () => {
  it("sorts a local version above the same version without one", () => {
    expectAscending(["1.0", "1.0+abc"]);
  });

  it("orders local segments part by part, numbers above text", () => {
    expectAscending(["1.0+a", "1.0+abc", "1.0+abc.5", "1.0+abc.7", "1.0+local", "1.0+1", "1.0+5"]);
  });

  it("lowercases local text and splits on any of . - _", () => {
    expectAllEqual(["1.0+abc.5", "1.0+ABC-5", "1.0+Abc_5"]);
  });
});

describe("compareVersions -- the legacy fallback", () => {
  // Most biocontainer build strings are not PEP 440 versions at all: parse_tag's "-1"
  // placeholder, conda build strings, mulled version hashes. packaging 21.3's LegacyVersion
  // hard-codes an epoch of -1 for those, and a PEP 440 epoch is never negative.
  const legacy = [
    "abc",
    "final",
    "-1",
    "h00cdaf9_0",
    "h9071d68_10",
    "hdbcaa40_3",
    "he860b03_2",
    "he941832_1",
    "py_1",
    "py27_0",
    "py35_0",
    "py36_0",
    "py311h1128e8f_0",
    "r43hc72bb7e_0",
    "6725cda82000b8e514baddcbf8c2dce054e3f797",
  ];

  it("orders the build strings biocontainer tags actually carry", () => {
    expectAscending(legacy);
  });

  it("puts every legacy version below every PEP 440 one", () => {
    for (const l of legacy) {
      for (const v of ["0", "0.0.1", "1.17", "1!0.1"]) {
        expect(compareVersions(l, v), `${l} < ${v}`).toBeLessThan(0);
      }
    }
  });

  it("sorts parse_tag's -1 placeholder below any real build string", () => {
    // Which is what makes "any matching image:version combination with a build number be
    // considered newer", as Galaxy's parse_tag comment puts it.
    expect(compareVersions("-1", "h00cdaf9_0")).toBeLessThan(0);
    expect(compareVersions("-1", "0")).toBeLessThan(0);
  });
});

describe("compareVersions -- numbers Python holds exactly", () => {
  // Every numeric component is a bigint. Python compares ints exactly; a JS number stops
  // being exact at 2^53, and two versions that compare equal are left in whatever order the
  // registry listed them, so the "newest" tag becomes a coin toss.
  it("separates the pair either side of 2^53", () => {
    expect(compareVersions("9007199254740992", "9007199254740993")).toBeLessThan(0);
    expect(compareVersions("9007199254740993", "9007199254740992")).toBeGreaterThan(0);
    expect(compareVersions("9007199254740993", "9007199254740993")).toBe(0);
    // The reason: as doubles they are one value.
    expect(9007199254740992 === 9007199254740993).toBe(true);
  });

  it("separates 30-digit release components", () => {
    const big = `1.${"9".repeat(30)}`;
    const smaller = `1.${"9".repeat(29)}8`;
    expect(compareVersions(smaller, big)).toBeLessThan(0);
    expect(compareVersions(big, smaller)).toBeGreaterThan(0);
    expect(compareVersions(big, big)).toBe(0);
    expect(Number(big) === Number(smaller)).toBe(true);
  });

  it("separates an epoch, a pre, a post, a dev and a local segment past 2^53", () => {
    expectAscending(["9007199254740992!1.0", "9007199254740993!1.0"]);
    expectAscending(["1.0a9007199254740992", "1.0a9007199254740993"]);
    expectAscending(["1.0.post9007199254740992", "1.0.post9007199254740993"]);
    expectAscending(["1.0.dev9007199254740992", "1.0.dev9007199254740993"]);
    expectAscending(["1.0+9007199254740992", "1.0+9007199254740993"]);
    // The bare -N post-release form, which shares the same parse.
    expectAscending(["1.0-9007199254740992", "1.0-9007199254740993"]);
  });

  it("keeps trailing-zero normalisation working on a huge release", () => {
    const huge = "1." + "9".repeat(30);
    expectAllEqual([huge, `${huge}.0`, `${huge}.0.0`]);
  });
});

describe("compareVersions -- Python's regex semantics, which JavaScript's are not", () => {
  // Every expectation here was read off the Galaxy checkout's own parse_version. The three
  // constructs packaging's pattern uses that have no JavaScript equivalent are `\s` (in the
  // anchors), `re.IGNORECASE` and, in the legacy splitter, `\d`.
  const NEL = "";
  const BOM = "﻿";
  const KELVIN = "K";
  const DOTTED_I = "İ";
  const DOTLESS_I = "ı";
  const LONG_S = "ſ";
  const ARABIC_FIVE = "٥";

  it("strips the whitespace Python strips, which includes U+0085", () => {
    // `^\s*` matches U+0085 and the C0 separators; String.prototype.trim does not, so this
    // used to fall through to the legacy key and sort below every PEP 440 version.
    expectAllEqual([`${NEL}1.0`, "1.0", `1.0 `, "1.0"]);
    expect(parseVersion(`${NEL}1.0`)).toMatchObject({ legacy: false, release: [1n] });
    expect(compareVersions(`${NEL}1.0`, "0.9")).toBeGreaterThan(0);
  });

  it("keeps the U+FEFF that Python keeps, so it stays a legacy version", () => {
    // The mirror image: trim removes a byte-order mark and quietly repaired `﻿1.0`
    // into the PEP 440 version 1.0, which Python never reads it as.
    expect(parseVersion(`${BOM}1.0`)).toMatchObject({ legacy: true, epoch: -1n });
    expect(compareVersions(`${BOM}1.0`, "0.9")).toBeLessThan(0);
  });

  it("folds the four non-ASCII letters Python's IGNORECASE folds, and no others", () => {
    // `[a-z]` under re.IGNORECASE matches the 52 ASCII letters plus these four. JavaScript's
    // `i` flag matches none of them and `iu` matches only two, so all four fell to the
    // legacy key -- below every PEP 440 version rather than equal to or just above `1.0`.
    expectAllEqual([`1.0+${KELVIN}`, "1.0+k", "1.0+K"]);
    for (const [version, ascii] of [
      [`1.0+${DOTTED_I}`, "1.0+i"],
      [`1.0+${DOTLESS_I}`, "1.0+i"],
      [`1.0+${LONG_S}`, "1.0+s"],
    ]) {
      expect(parseVersion(version!)).toMatchObject({ legacy: false });
      // Not equal: the local segment is lowercased, and only the Kelvin sign lowercases
      // onto an ASCII letter. It sorts above its ASCII neighbour, as it does in Python.
      expect(compareVersions(version!, ascii!)).toBeGreaterThan(0);
    }
  });

  it("carries a folded letter into the pre/post spelling, as Python does", () => {
    // "poſt" matches `post` under IGNORECASE but `.lower()` leaves it as it is, so it is not
    // the alias `post` and sorts above it on the letter.
    expect(parseVersion(`1.0.po${LONG_S}t1`)).toMatchObject({ legacy: false });
    expect(compareVersions("1.0.post1", `1.0.po${LONG_S}t1`)).toBeLessThan(0);
  });

  it("splits a legacy version on Python's `\\d`, which is every Nd digit", () => {
    // Python reads `٥5` as one numeric run and stars it whole, because the zero-padding
    // test right after the split is ASCII-only. JavaScript's `\d` is ASCII, so it used to
    // split that run in two and zero-pad the second half into a numeric part -- which then
    // compares as a number where Python compares the text.
    expect(parseVersion(`a${ARABIC_FIVE}5`)).toMatchObject({
      legacy: true,
      parts: ["*a", `*${ARABIC_FIVE}5`, "*final"],
    });
    expect(parseVersion(`a${ARABIC_FIVE}z`)).toMatchObject({
      legacy: true,
      parts: ["*a", `*${ARABIC_FIVE}`, "*z", "*final"],
    });
    expect(compareVersions(`a${ARABIC_FIVE}5`, `a${ARABIC_FIVE}10`)).toBeGreaterThan(0);
  });

  it("orders legacy parts by code point, not by UTF-16 code unit", () => {
    // U+E000 is below U+10000 for Python and above it for JavaScript, because an astral
    // character starts with a surrogate at U+D800.
    expect(compareVersions("x", "x\u{10000}")).toBeLessThan(0);
    expect("x" < "x\u{10000}").toBe(false);
  });
});

describe("parseVersion", () => {
  it("marks a PEP 440 version and a legacy one apart by epoch, as packaging does", () => {
    expect(parseVersion("1.0")).toMatchObject({ legacy: false, epoch: 0n, release: [1n] });
    expect(parseVersion("h00cdaf9_0")).toMatchObject({ legacy: true, epoch: -1n });
  });

  it("gives a key that can be compared directly, so a sort parses each tag once", () => {
    const [a, b] = [parseVersion("1.17"), parseVersion("1.17.1")];
    expect(compareVersionKeys(a, b)).toBeLessThan(0);
    expect(compareVersionKeys(b, a)).toBeGreaterThan(0);
    expect(compareVersionKeys(a, a)).toBe(0);
  });
});

describe("the legacy key's zero padding", () => {
  it("pads a numeric run by code points, so an astral digit does not shift the comparison", () => {
    // MATHEMATICAL DOUBLE-STRUCK DIGIT ZERO is Nd, so `_legacy_version_component_re`
    // keeps it inside the numeric run "1𝟘" -- and `zfill(8)` pads that to eight code points where
    // `padStart(8)` pads to eight UTF-16 units, which is one "0" fewer and puts the "1" a place
    // to the left. Python: sorted(["a1𝟘", "a20"], key=parse_version) == ["a1𝟘", "a20"], i.e.
    // a20 is the newer tag.
    expect(compareVersions("a1\u{1D7D8}", "a20")).toBeLessThan(0);
    expect(parseVersion("a1\u{1D7D8}")).toMatchObject({
      legacy: true,
      parts: ["*a", "0000001\u{1D7D8}", "*final"],
    });
    expect(parseVersion("a20")).toMatchObject({
      legacy: true,
      parts: ["*a", "00000020", "*final"],
    });
  });
});

describe("int()'s digit limit, which every number in the parse crosses the same way", () => {
  // `Version.__init__` builds its key with `int()` five times -- epoch, each release component,
  // a pre / post / dev numeral, the implicit post-release numeral, and a numeric local part --
  // and `int()` refuses a string of more than 4300 digits. The ValueError it raises is not an
  // InvalidVersion, so `parse_version` does not fall back to the legacy key: it propagates, out
  // of `version_sorted` and out of `recommend_container`. Every row below is
  // `galaxy.tool_util.version.parse_version` on the reference interpreter.
  const nines = (n: number) => "9".repeat(n);

  it.each([
    ["a release", nines(4301)],
    ["a release component after the first", `1.${nines(4301)}`],
    ["an epoch", `${nines(4301)}!1.0`],
    ["a pre-release numeral", `1.0a${nines(4301)}`],
    ["the implicit post-release numeral", `1.0-${nines(4301)}`],
    ["a dev-release numeral", `1.0.dev${nines(4301)}`],
    ["a numeric local part", `1.0+${nines(4301)}`],
  ])("refuses 4301 digits in %s, as int() does", (_what, version) => {
    expect(() => parseVersion(version)).toThrow(GalaxyValidationError);
    expect(() => parseVersion(version)).toThrow(
      "Exceeds the limit (4300 digits) for integer string conversion: value has 4301 digits",
    );
  });

  it.each([
    ["a release", nines(4300), { release: [BigInt(nines(4300))] }],
    ["an epoch", `${nines(4300)}!1.0`, { epoch: BigInt(nines(4300)) }],
    ["a numeric local part", `1.0+${nines(4300)}`, { local: [{ num: BigInt(nines(4300)) }] }],
  ])("reads exactly 4300 digits in %s, as int() does", (_what, version, key) => {
    expect(parseVersion(version)).toMatchObject({ legacy: false, ...key });
  });

  it("leaves the legacy fallback alone, because its key never calls int()", () => {
    // `_parse_version_parts` zero-pads a numeric run as a STRING, so a run this long is a legacy
    // key in Python too rather than a ValueError: parse_version("a_" + 4301 nines) is a
    // LegacyVersion there.
    expect(parseVersion(`a_${nines(4301)}`)).toMatchObject({ legacy: true, epoch: -1n });
  });
});
