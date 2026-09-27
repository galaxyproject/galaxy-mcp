/**
 * The handful of Python string semantics a faithful port of Galaxy's recommender needs, in
 * one place because two modules need them and they cannot both own them: `mulled.ts` strips
 * and orders caller input, `pep440.ts` strips and orders inside the version parser, and
 * `mulled.ts` already imports `pep440.ts`.
 *
 * Everything here is `str`'s, not JavaScript's, and each one differs from the JavaScript
 * spelling it looks like in a way that changes which container gets recommended.
 */
import { GalaxyValidationError } from "./errors";

/**
 * The code points CPython's `str.isspace()` is true for -- bidirectional class WS, B or S, or
 * category Zs, which is what `_PyUnicode_IsWhitespace` tests. That set is what `str.strip()`
 * removes and what a str-pattern `\s` matches, and it was read off the installed interpreter
 * over the whole code space rather than recalled.
 *
 * It is not JavaScript's. `String.prototype.trim` takes Unicode's White_Space property, drops
 * U+0085 and adds U+FEFF, so the two disagree on five code points -- and both directions change
 * answers here, because a name and a version are stripped before they are validated, hashed
 * and matched against a tag:
 *
 *  - U+0085 and the four C0 separators U+001C-U+001F: Python strips them, `trim` keeps them,
 *    so `samtools=1.17` resolves there and would be refused here as an unsafe version.
 *  - U+FEFF: `trim` strips it, Python keeps it, so `samtools=﻿1.17` is a version Python
 *    never finds a tag for -- a name_only answer with a note -- which `trim` would silently
 *    repair into an exact_version match.
 *
 * `laxen.ts` has `trimPydanticSpace` for the same job on the request-coercion path, and it is
 * deliberately not reused: that one is pydantic's, i.e. Rust's `char::is_whitespace`, which is
 * exactly the White_Space property and so lacks U+001C-U+001F. One set is a Python string
 * method and the other a Rust one; they differ by those four separators.
 */
export function isPySpace(code: number): boolean {
  return (
    (code >= 0x09 && code <= 0x0d) ||
    // The C0 information separators, and then U+0020 -- one contiguous run.
    (code >= 0x1c && code <= 0x20) ||
    code === 0x85 ||
    code === 0xa0 ||
    code === 0x1680 ||
    (code >= 0x2000 && code <= 0x200a) ||
    code === 0x2028 ||
    code === 0x2029 ||
    code === 0x202f ||
    code === 0x205f ||
    code === 0x3000
  );
}

/**
 * Python's `str.strip()`. Two scans and a slice rather than a regex, as laxen's trim is, so a
 * long run of whitespace stays linear.
 */
export function pyStrip(value: string): string {
  let start = 0;
  let end = value.length;
  while (start < end && isPySpace(value.charCodeAt(start))) start += 1;
  while (end > start && isPySpace(value.charCodeAt(end - 1))) end -= 1;
  return start === 0 && end === value.length ? value : value.slice(start, end);
}

/**
 * Python's `<` on two `str`s: code point by code point, then the shorter string first.
 *
 * JavaScript's `<` compares UTF-16 code units, and the two orders disagree for every pair
 * where one string has an astral character (U+10000 and up, stored as a surrogate pair
 * starting at U+D800) and the other has a BMP character above U+DFFF. U+E000 sorts below
 * U+10000 for Python and above it for JavaScript.
 *
 * That is not academic here: `v2_image_name` hashes the package names **in sorted order** and
 * the recommendation cache keys on sorted `(name, version)` pairs, so a set of names the two
 * languages order differently hashes to a different mulled-v2 repository and asks quay.io
 * about a different image. The version comparator has the same exposure through the legacy
 * key, whose parts are arbitrary slices of a tag name.
 *
 * Iterating with `for...of` walks code points, and yields a lone surrogate as itself -- which
 * is what Python does with the lone surrogate `json.loads` builds from a `\uD800` escape.
 */
export function comparePyStrings(a: string, b: string): number {
  if (a === b) return 0;
  const ai = a[Symbol.iterator]();
  const bi = b[Symbol.iterator]();
  for (;;) {
    const x = ai.next();
    const y = bi.next();
    if (x.done === true) return y.done === true ? 0 : -1;
    if (y.done === true) return 1;
    const cx = x.value.codePointAt(0)!;
    const cy = y.value.codePointAt(0)!;
    if (cx !== cy) return cx < cy ? -1 : 1;
  }
}

/** Python's `len()`: code points, counting a lone surrogate as one, as the iterator yields them. */
function pyLen(value: string): number {
  let n = 0;
  for (let i = 0; i < value.length; i += 1) {
    const c = value.charCodeAt(i);
    if (c >= 0xd800 && c <= 0xdbff && i + 1 < value.length) {
      const low = value.charCodeAt(i + 1);
      // A well-formed pair is one code point; a lone surrogate is one too, as in Python.
      if (low >= 0xdc00 && low <= 0xdfff) i += 1;
    }
    n += 1;
  }
  return n;
}

/**
 * Python's `str.zfill(width)`: pad with "0" on the left to `width` CODE POINTS, after a sign.
 *
 * `padStart` counts UTF-16 code units, so a string carrying an astral character comes out a code
 * point short of what `zfill` produces -- and `_legacy_cmpkey`'s padded parts are then compared
 * as strings, where one missing pad character shifts every digit one place. `"1\u{1D7D8}"`
 * padded by `padStart(8, "0")` is `"000001𝟘"` and by `zfill(8)` is `"0000001𝟘"`, which is why
 * Python orders `a1𝟘` below `a20` and `padStart` orders it above.
 *
 * The sign rule is `zfill`'s: the fill goes after a leading `+` or `-`, so `"-1".zfill(4)` is
 * `"-001"`. Nothing here reaches it -- `_parse_version_parts` only pads a part that starts with
 * an ASCII digit -- but a helper named after `zfill` should be `zfill`.
 */
export function pyZfill(value: string, width: number): string {
  const n = pyLen(value);
  if (n >= width) return value;
  const signed = value.startsWith("+") || value.startsWith("-") ? 1 : 0;
  return value.slice(0, signed) + "0".repeat(width - n) + value.slice(signed);
}

/** `\p{Nd}`, which is what a Python str pattern's `\d` matches, for one code point. */
const IS_ND = /^\p{Nd}$/u;

/**
 * `sys.get_int_max_str_digits()`, whose default CPython 3.11 and later ship is 4300.
 *
 * (Not `sys.int_info.str_digits_check_threshold`, which is 640 -- that is the lowest value the
 * limit may be set to, not the limit.)
 */
export const PY_INT_MAX_STR_DIGITS = 4300;

/**
 * Python's `int()` over the digits a `\d+` match holds -- the whole Nd category, not `[0-9]`.
 *
 * `BigInt("٢")` throws where `int("٢")` is 2, because every Nd code point carries a decimal
 * value. Unicode lays each decimal digit set out as ten ascending code points starting with its
 * zero, so a digit's value is its offset inside the aligned block that holds it; adjacent sets
 * make runs longer than ten -- the five mathematical-digit sets at U+1D7CE..U+1D7FF are one run
 * of fifty -- which is why the offset is taken modulo ten rather than from the run's start.
 * Checked against `int()` for all 680 code points the reference interpreter calls Nd: no
 * disagreement.
 *
 * `int()` also REFUSES a long enough string, and `BigInt` has no such limit: over
 * `sys.get_int_max_str_digits()` digits raises a ValueError rather than converting, so a tag
 * whose build number is 4301 nines makes `parse_tag` raise where this used to hand back a
 * number and recommend the tag. The refusal counts digit characters as written, leading zeros
 * and all -- `int("0" * 4300 + "9")` reports 4301 digits -- and counts a code point once, so an
 * astral digit counts as one. CPython's message, and `GalaxyValidationError` because that is
 * this surface's spelling of the ValueError it is: it leaves the recommender uncaught (Python's
 * `except RequestException` does not cover it) and `recommend_biocontainer` re-raises it as a
 * ValueError of its own, which is the same path `build_target`'s refusals take. The digits come
 * from a registry tag rather than from the caller, which is unusual for that class, and is why
 * the message says what the limit is.
 */
export function pyIntFromDigits(digits: string): bigint {
  const count = pyLen(digits);
  if (count > PY_INT_MAX_STR_DIGITS) {
    throw new GalaxyValidationError(
      `Exceeds the limit (${PY_INT_MAX_STR_DIGITS} digits) for integer string conversion: ` +
        `value has ${count} digits; use sys.set_int_max_str_digits() to increase the limit`,
    );
  }
  let decimal = "";
  for (const ch of digits) {
    const c = ch.codePointAt(0)!;
    if (c >= 0x30 && c <= 0x39) {
      decimal += ch;
      continue;
    }
    if (!IS_ND.test(ch)) throw new RangeError(`not a decimal digit: ${pyRepr(ch)}`);
    let zero = c;
    while (zero > 0 && IS_ND.test(String.fromCodePoint(zero - 1))) zero -= 1;
    decimal += String((c - zero) % 10);
  }
  // `int("")` is a ValueError there and `BigInt("")` is 0n here, so it is not a silent 0.
  if (!decimal) throw new RangeError("no digits to convert");
  return BigInt(decimal);
}

/**
 * The code points `str.isprintable()` is false for: categories Cc, Cf, Cs, Co, Cn, Zl, Zp and
 * Zs, minus U+0020, which is what `Py_UNICODE_ISPRINTABLE` tests.
 *
 * Checked against the interpreter over the whole code space: it agrees on every code point
 * Python 3.12 calls assigned, and differs only where Node's newer Unicode has assigned
 * something Python still calls Cn -- the same version skew `\p{Nd}` carries for `\d`.
 */
const PY_NONPRINTABLE = /[\p{Cc}\p{Cf}\p{Cs}\p{Co}\p{Cn}\p{Zl}\p{Zp}\p{Zs}]/u;

const hex = (c: number, width: number) => c.toString(16).padStart(width, "0");

/**
 * Python's `repr()` of a `str`, which is what an f-string interpolating a container renders its
 * elements with -- so it is part of the note `_recommend_multi` returns, not just a debug aid.
 *
 * `unicode_repr` in CPython, condition for condition: the quote is `'` unless the string
 * contains one and no `"`; the chosen quote and `\` are backslash-escaped; tab, newline and
 * carriage return get their short escapes; anything below U+0020 or U+007F is `\xhh`; the rest
 * of ASCII is itself; above that, a printable code point is itself and a non-printable one is
 * `\xhh`, `\uhhhh` or `\Uhhhhhhhh` by width, in lowercase hex.
 */
export function pyRepr(value: string): string {
  const quote = value.includes("'") && !value.includes('"') ? '"' : "'";
  let out = quote;
  for (const ch of value) {
    const c = ch.codePointAt(0)!;
    if (ch === quote || ch === "\\") out += `\\${ch}`;
    else if (ch === "\t") out += "\\t";
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (c < 0x20 || c === 0x7f) out += `\\x${hex(c, 2)}`;
    else if (c < 0x7f) out += ch;
    else if (!PY_NONPRINTABLE.test(ch)) out += ch;
    else if (c < 0x100) out += `\\x${hex(c, 2)}`;
    else if (c < 0x10000) out += `\\u${hex(c, 4)}`;
    else out += `\\U${hex(c, 8)}`;
  }
  return out + quote;
}

/** Python's `repr()` of a `list[str]`, which is `[` plus the element reprs joined with `, `. */
export const pyReprList = (values: readonly string[]): string =>
  `[${values.map((v) => pyRepr(v)).join(", ")}]`;

/**
 * The `UnicodeEncodeError` message `value.encode("utf-8")` would raise, or null when it would
 * not raise at all.
 *
 * Python refuses to encode a surrogate code point: a name carrying one -- which is what
 * `json.loads` builds from a `"\uD800"` escape, and what a CLI argument can carry -- fails
 * before a byte of it reaches a hash or a URL. JavaScript has no such refusal; both
 * `TextEncoder` and `createHash(...).update(s, "utf8")` substitute U+FFFD and carry on, which
 * hashes a different string than Python would have hashed if Python had got that far.
 *
 * The message is CPython's, including its two shapes: one unencodable code point names itself,
 * a run of them reports the range. Positions are code point indices, as Python counts them, so
 * a well-formed surrogate pair is one position and is encodable.
 */
export function pyUtf8EncodeError(value: string): string | null {
  let pos = -1;
  let start = -1;
  let end = -1;
  let first = "";
  for (const ch of value) {
    pos += 1;
    const c = ch.codePointAt(0)!;
    if (c >= 0xd800 && c <= 0xdfff) {
      if (start < 0) {
        start = pos;
        first = ch;
      }
      end = pos;
    } else if (start >= 0) {
      break;
    }
  }
  if (start < 0) return null;
  const where =
    start === end
      ? `character ${pyRepr(first)} in position ${start}`
      : `characters in position ${start}-${end}`;
  return `'utf-8' codec can't encode ${where}: surrogates not allowed`;
}

/**
 * The character class Python's `\w` is, on a `str` -- spelled out, because JavaScript's is not.
 *
 * `re` on a `str` matches `\w` against a letter, a number or an underscore in any script, and
 * that set is exactly Unicode's general categories L and N plus U+005F: the installed
 * interpreter was asked about every code point in the space and it disagreed with
 * `[\p{L}\p{N}_]` on none of them, in either direction. JavaScript's `\w` is `[A-Za-z0-9_]` and
 * nothing else, so a port that leaves it alone reads a Greek or accented letter as a separator.
 *
 * The class matters most where the pattern never mentions it: `\b`. A boundary is a place where
 * one side is a word character and the other is not, so `\b` inherits whichever `\w` the engine
 * has -- which is why `re.findall(r"\b[a-zA-Z]{2,}\b", "café")` finds nothing (the `é`
 * continues the word) while the same pattern in JavaScript finds `caf`. JavaScript has no
 * Unicode `\b` to switch on, so a boundary has to be built out of this class with lookaround,
 * and the pattern needs the `u` flag or the class stops at a lone surrogate.
 *
 * One residue that no spelling fixes: this is a Unicode *data* question, and the two runtimes
 * carry different editions of the data. Node 22's is 17.0 and the interpreter here reports
 * 15.0.0, which knows 9,661 fewer code points as letters or numbers -- U+A7CB and the Garay
 * block among them. Those are word characters here and separators there until the interpreter
 * updates, at which point the two agree again; everything assigned before Unicode 15.0.0, which
 * is every character a Galaxy name or an IWC readme has carried so far, matches today.
 */
export const PY_WORD_CHAR = "[\\p{L}\\p{N}_]";
