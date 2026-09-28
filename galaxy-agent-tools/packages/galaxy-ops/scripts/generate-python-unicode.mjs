/**
 * Turn the other server's checked-in Unicode tables into the module this package reads.
 *
 * Run from packages/galaxy-ops:
 *
 *   node scripts/generate-python-unicode.mjs
 *
 * The input is mcp-server-galaxy-py/tests/testdata/python-unicode-<version>.json, which the
 * Python suite generates off its own interpreter (`uv run python -m tests.python_unicode`).
 * The output is src/python-unicode-data.ts, checked in beside the code that uses it, so the
 * package has no build step that needs a Python and no runtime dependency on this runtime's
 * Unicode edition.
 *
 * The word-character ranges are copied across as they are. The lowercasing table is the
 * short list of code points where this runtime's `toLowerCase` disagrees with the other
 * server's `str.lower()` -- computed here, because it is a fact about both sides -- and
 * `test/python-unicode-data.test.ts` recomputes it over the whole code space, so a Node
 * whose Unicode data has moved on fails there rather than diverging quietly.
 */
import { readFileSync, writeFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";

const TESTDATA = new URL("../../../../mcp-server-galaxy-py/tests/testdata/", import.meta.url);
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

/** The code points the other server's `lower()` leaves alone and this runtime's does not. */
const lowered = new Map(data.lower_map.rows);
const exceptions = [];
for (let cp = 0; cp < 0x110000; cp += 1) {
  const character = String.fromCodePoint(cp);
  const theirs = lowered.get(cp) ?? character;
  if (theirs !== character.toLowerCase()) exceptions.push(cp);
}
const exceptionRanges = [];
for (const cp of exceptions) {
  const last = exceptionRanges[exceptionRanges.length - 1];
  if (last && last[1] === cp - 1) last[1] = cp;
  else exceptionRanges.push([cp, cp]);
}

const text = `/**
 * The Unicode facts the other server's string methods are built on, copied from that
 * server's checked-in tables rather than read off this runtime's.
 *
 * GENERATED FILE -- do not edit. Regenerate from packages/galaxy-ops with
 * \`node scripts/generate-python-unicode.mjs\`. Its input is
 * mcp-server-galaxy-py/tests/testdata/${source}, which that server writes with
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
 * Inclusive ranges of the code points this runtime lowercases and the other server does
 * not: ${exceptions.length} of them, each one a character Unicode gave a case mapping
 * after ${data.unicode_version}. \`pyLower\` leaves exactly these alone.
 */
export const PY_LOWER_EXCEPTION_RANGES: readonly (readonly [number, number])[] = [
${rangeLines(exceptionRanges)}
];
`;

writeFileSync(fileURLToPath(OUT), text);
console.log(
  `wrote src/python-unicode-data.ts from ${source}: ${data.word_chars.count} word-char ranges, ` +
    `${exceptions.length} lowercasing exceptions`,
);
