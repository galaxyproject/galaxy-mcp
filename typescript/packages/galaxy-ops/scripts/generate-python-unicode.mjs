/**
 * Turn the other server's checked-in Unicode tables into the module this package reads.
 *
 * Run from packages/galaxy-ops:
 *
 *   node scripts/generate-python-unicode.mjs
 *
 * The input is python/tests/testdata/python-unicode-<version>.json, which the
 * Python suite generates off its own interpreter (`uv run python -m tests.python_unicode`).
 * The output is src/python-unicode-data.ts, checked in beside the code that uses it, so the
 * package has no build step that needs a Python and no runtime dependency on this runtime's
 * Unicode edition.
 *
 * Everything is copied across as it is written down there -- the word-character ranges, the
 * whole one-code-point lowercase mapping, and the two sets the final-sigma rule reads. Nothing
 * here asks this runtime anything about Unicode, and `test/python-unicode-data.test.ts`
 * compares both copies row for row so they cannot drift.
 */
import { readFileSync, writeFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";

const TESTDATA = new URL("../../../../contract/python-unicode/", import.meta.url);
const OUT = new URL("../src/python-unicode-data.ts", import.meta.url);

const names = readdirSync(fileURLToPath(TESTDATA)).filter(
  (n) => n.startsWith("python-unicode-") && n.endsWith(".json"),
);
if (names.length !== 1) {
  throw new Error(
    `expected exactly one python-unicode-*.json in ${fileURLToPath(TESTDATA)}, found ` +
      `${names.length} (${names.join(", ")}) -- the pinned tables are one interpreter's`,
  );
}
const source = names[0];
const data = JSON.parse(readFileSync(new URL(source, TESTDATA), "utf8"));

const hex = (cp) => `0x${cp.toString(16).padStart(4, "0")}`;
const rangeLines = (rows) => rows.map(([lo, hi]) => `  [${hex(lo)}, ${hex(hi)}],`).join("\n");

/** A string literal for the code points `lower()` makes of one character. */
const literal = (codePoints) => {
  const body = codePoints
    .map((cp) => {
      if (cp === 0x22 || cp === 0x5c) return `\\${String.fromCodePoint(cp)}`;
      if (cp >= 0x20 && cp <= 0x7e) return String.fromCodePoint(cp);
      return cp > 0xffff ? `\\u{${cp.toString(16)}}` : `\\u${cp.toString(16).padStart(4, "0")}`;
    })
    .join("");
  return `"${body}"`;
};
const lowerLines = (rows) =>
  rows.map(([cp, lowered]) => `  [${hex(cp)}, ${literal(lowered)}],`).join("\n");

const text = `/**
 * The Unicode facts the other server's string methods are built on, copied from that
 * server's checked-in tables rather than read off this runtime's.
 *
 * GENERATED FILE -- do not edit. Regenerate from packages/galaxy-ops with
 * \`node scripts/generate-python-unicode.mjs\`. Its input is
 * python/tests/testdata/${source}, which that server writes with
 * \`uv run python -m tests.python_unicode\`, and \`test/python-unicode-data.test.ts\`
 * compares this file to that one row for row, so the two copies cannot drift.
 *
 * Why a table at all: a word boundary, the whitespace a summary is split on and a
 * lowercased needle are all decided by Unicode data, and the two runtimes carry different
 * editions of it. This one knows thousands of code points as letters that Unicode
 * ${data.unicode_version} had not assigned yet, so \`\\p{L}\` here and \`\\w\` there
 * answer differently about them -- and an intent of "rnaseq" followed by one of those
 * letters is a searchable term on one server and no term at all on the other. The other
 * server is the contract, so its tables are too.
 */

/** The Unicode edition these tables are, which is the one the contract interpreter carries. */
export const PY_UNICODE_VERSION = "${data.unicode_version}";

/**
 * Inclusive code point ranges of the set \`\\w\` matches on a \`str\`, which is the set a
 * \`\\b\` in that engine reads: ${data.word_chars.count} ranges holding
 * ${data.word_chars.code_points.toLocaleString("en-US")} code points.
 */
export const PY_WORD_CHAR_RANGES: readonly (readonly [number, number])[] = [
${rangeLines(data.word_chars.rows)}
];

/**
 * The cased characters, as the final-sigma rule asks about them: ${data.cased.count} ranges
 * holding ${data.cased.code_points.toLocaleString("en-US")} code points. A sigma is a final
 * sigma when one of these is behind it and none is in front, with only case-ignorable
 * characters in between either way.
 *
 * "As the rule asks about them" is the whole of the subtlety: the interpreter walks past
 * case-ignorable characters before it asks whether anything is cased, so a character that is
 * both -- U+0345 among them -- never gets asked, and this set leaves those out.
 */
export const PY_CASED_RANGES: readonly (readonly [number, number])[] = [
${rangeLines(data.cased.rows)}
];

/**
 * The characters the final-sigma rule walks past on its way to a cased one, in either
 * direction: ${data.case_ignorable.count} ranges holding
 * ${data.case_ignorable.code_points.toLocaleString("en-US")} code points -- accents, the soft
 * hyphen, quotation marks, modifier letters.
 */
export const PY_CASE_IGNORABLE_RANGES: readonly (readonly [number, number])[] = [
${rangeLines(data.case_ignorable.rows)}
];

/**
 * Every code point the contract interpreter's \`str.lower()\` does not leave alone, and what
 * it makes of it: ${data.lower_map.count.toLocaleString("en-US")} of them, the complete
 * mapping rather than the places this runtime disagrees. U+0130 maps to two code points; the
 * rest map to one. U+03A3 is in here as its plain lowercase, and the rule that overrides it in
 * context reads the two sets above.
 */
export const PY_LOWER_MAP: readonly (readonly [number, string])[] = [
${lowerLines(data.lower_map.rows)}
];
`;

writeFileSync(fileURLToPath(OUT), text);
console.log(
  `wrote src/python-unicode-data.ts from ${source}: ${data.word_chars.count} word-char ranges, ` +
    `${data.cased.count} cased ranges, ${data.case_ignorable.count} case-ignorable ranges, ` +
    `${data.lower_map.count} lowercase mappings`,
);
