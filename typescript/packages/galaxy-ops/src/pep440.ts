/**
 * Version ordering as Galaxy sees it, so a tag list sorts the same on both surfaces.
 *
 * Galaxy's `version_sorted` orders tags by `galaxy.tool_util.version.parse_version`, which is
 * `packaging.version.Version` with a fallback to the `LegacyVersion` that packaging 21.3
 * carried before it was removed (Galaxy vendored that class into
 * `lib/galaxy/tool_util/version.py`). Both are ported here: the PEP 440 grammar and
 * `packaging._cmpkey`'s ordering rules for epoch / release / pre / post / dev / local, and
 * the setuptools-era legacy key for everything that is not a PEP 440 version -- which is most
 * biocontainer build strings (`h00cdaf9_0`, `py311h1128e8f_0`, the `-1` placeholder
 * `parse_tag` uses when a tag has no build at all).
 *
 * The two live in one ordering: a legacy key's epoch is hard-coded to -1 and a PEP 440
 * epoch is never negative, so anything that does not parse sorts below everything that does,
 * exactly as it does in Python.
 *
 * Sources: `packaging/version.py` (`VERSION_PATTERN`, `_parse_letter_version`,
 * `_parse_local_version`, `_cmpkey`) and `galaxy/tool_util/version.py`
 * (`parse_version`, `_legacy_cmpkey`).
 *
 * Both are Python regular expressions over Python strings, and three of those semantics have
 * no JavaScript equivalent that can be reached by writing the pattern out verbatim: `\s`,
 * `\d`, and `re.IGNORECASE`. Each is spelled explicitly below, with the code points it covers
 * read off the installed interpreter rather than recalled.
 */
import { comparePyStrings, pyIntFromDigits, pyStrip, pyZfill } from "./python-str";

/**
 * Every `int()` in packaging's parse, aliased so no conversion here can quietly skip the check.
 *
 * There are five: the epoch, each release component, a pre / post / dev numeral, the implicit
 * post-release numeral, and a numeric local-version part. `int()` refuses a string of more than
 * `sys.get_int_max_str_digits()` digits and `BigInt` has no such limit, so a tag whose release
 * is 4301 nines (`4301-nines--h0_0`) raises a ValueError out of `parse_version` there -- which
 * `version_sorted` does not catch and `recommend_container` re-raises -- and used to parse and
 * be recommended here. The digits reaching these five are `[0-9]` by the pattern above them, so
 * what is wanted from `pyIntFromDigits` is the limit rather than its Nd handling.
 */
const pyInt = pyIntFromDigits;

/**
 * An `Infinity` / `-Infinity` sentinel or a `(letter, number)` pair, as `_cmpkey` builds.
 *
 * Every numeric component in this file is a bigint, not a number. Python holds these as ints
 * and compares them exactly; a JS number stops being exact at 2^53, where `9007199254740992`
 * and `9007199254740993` are the same value -- so a comparator built on numbers reports two
 * different versions equal, `version_sorted` leaves them in whichever order quay.io listed
 * them, and the "newest" tag is a coin toss. Both are legal PEP 440 releases.
 */
type Marker =
  | { readonly kind: "-inf" }
  | { readonly kind: "inf" }
  | { readonly kind: "pair"; readonly letter: string; readonly n: bigint };

const NEG_INF: Marker = { kind: "-inf" };
const POS_INF: Marker = { kind: "inf" };

/**
 * One element of the local-version key: packaging renders an integer part as `(i, "")` and a
 * string part as `(NegativeInfinity, i)`, which is how numeric parts sort above alphanumeric
 * ones. `num === null` is that NegativeInfinity.
 */
interface LocalPart {
  readonly num: bigint | null;
  readonly str: string;
}

export type VersionKey =
  | { readonly legacy: true; readonly epoch: bigint; readonly parts: readonly string[] }
  | {
      readonly legacy: false;
      readonly epoch: bigint;
      readonly release: readonly bigint[];
      readonly pre: Marker;
      readonly post: Marker;
      readonly dev: Marker;
      readonly local: readonly LocalPart[] | null;
    };

// ---------------------------------------------------------------------------
// PEP 440
// ---------------------------------------------------------------------------

/**
 * The extra code points Python's `re.IGNORECASE` folds onto an ASCII letter, beyond that
 * letter's own two cases. There are exactly four across the whole alphabet, and the
 * interpreter names them: `[a-z]` under IGNORECASE matches the 52 ASCII letters plus U+0130
 * LATIN CAPITAL LETTER I WITH DOT ABOVE, U+0131 LATIN SMALL LETTER DOTLESS I, U+017F LATIN
 * SMALL LETTER LONG S and U+212A KELVIN SIGN.
 */
const PY_CASE_EXTRAS: Record<string, string> = {
  i: "\\u0130\\u0131",
  k: "\\u212a",
  s: "\\u017f",
};

/**
 * One ASCII letter as the character class Python's `re.IGNORECASE` matches it with.
 *
 * JavaScript's `i` flag is not the same relation and neither is `iu`. Without `u`, case
 * folding is ASCII-ish and refuses U+212A and U+017F; with `u` it is simple case folding,
 * which accepts those two and still refuses U+0130 and U+0131, because neither folds to `i`.
 * Python accepts all four. Spelling the class out is the only way to get Python's set, and it
 * is a small set: it only ever differs for `i`, `k` and `s`.
 */
const ci = (literal: string) =>
  literal.replace(/[a-z]/g, (c) => `[${c}${c.toUpperCase()}${PY_CASE_EXTRAS[c] ?? ""}]`);

/** `[a-z0-9]` under IGNORECASE, which is the local-version alphabet. */
const CI_ALNUM = "[0-9a-zA-Z\\u0130\\u0131\\u017f\\u212a]";

/**
 * packaging's `VERSION_PATTERN`, laid out segment by segment the way its `re.VERBOSE`
 * source is. The outer `pre` / `post` / `dev` groups are plain groups here because only
 * their parts are ever read, and the pre-release spellings are longest-first so the
 * alternation does not need to backtrack through `a` to reach `alpha`.
 *
 * `Version._regex` wraps this in `^\s*` and `\s*$`, which are not here: `pep440Key` runs
 * `pyStrip` over the input instead, because a Python `\s` and a JavaScript one are different
 * sets and neither contains the other. The two are equivalent -- nothing the pattern can
 * match begins or ends with whitespace, and `$` matching before a trailing newline is moot
 * once the newline is stripped -- and it keeps one definition of Python's whitespace.
 */
const PEP440 = new RegExp(
  [
    `^${ci("v")}?(?:`,
    "(?:(?<epoch>[0-9]+)!)?", // epoch
    "(?<release>[0-9]+(?:\\.[0-9]+)*)", // release segment
    `(?:[-_.]?(?<pre_l>${ci("alpha|a|beta|b|preview|pre|c|rc")})[-_.]?(?<pre_n>[0-9]+)?)?`, // pre
    "(?:(?:-(?<post_n1>[0-9]+))|", // post, implicit -N form
    `(?:[-_.]?(?<post_l>${ci("post|rev|r")})[-_.]?(?<post_n2>[0-9]+)?))?`, // post, explicit
    `(?:[-_.]?(?<dev_l>${ci("dev")})[-_.]?(?<dev_n>[0-9]+)?)?`, // dev
    `)(?:\\+(?<local>${CI_ALNUM}+(?:[-_.]${CI_ALNUM}+)*))?$`, // local version
  ].join(""),
  "u",
);

/**
 * packaging's `_parse_letter_version`, spellings normalised the same way. `toLowerCase` is
 * `str.lower()`: both apply Unicode's full, locale-independent lowercase mapping including
 * the final-sigma rule, and they agree on every code point the older of the two runtimes'
 * Unicode tables knows about. It is not `toLocaleLowerCase`, which would be Turkish `i` under
 * a Turkish locale.
 */
function letterVersion(
  letter: string | undefined,
  number: string | undefined,
): { letter: string; n: bigint } | null {
  if (letter) {
    let l = letter.toLowerCase();
    if (l === "alpha") l = "a";
    else if (l === "beta") l = "b";
    else if (l === "c" || l === "pre" || l === "preview") l = "rc";
    else if (l === "rev" || l === "r") l = "post";
    // An absent numeral is an implicit 0.
    return { letter: l, n: number === undefined ? 0n : pyInt(number) };
  }
  // A number with no letter is the implicit post-release syntax, e.g. "1.0-1".
  if (number) return { letter: "post", n: pyInt(number) };
  return null;
}

/**
 * packaging's `_parse_local_version`: split on `[._-]`, digits numeric, the rest lowercased.
 * Python's test is `part.isdigit()`, which is wider than `[0-9]+` -- but the local alphabet
 * the pattern above allows through is ASCII alphanumerics plus four letters, so the two
 * decide every part that can reach here the same way.
 */
function parseLocal(local: string | undefined): LocalPart[] | null {
  if (local === undefined) return null;
  return local
    .split(/[._-]/)
    .map((part) =>
      /^[0-9]+$/.test(part)
        ? { num: pyInt(part), str: "" }
        : { num: null, str: part.toLowerCase() },
    );
}

function pep440Key(version: string): VersionKey | null {
  // `Version._regex`'s `^\s*` / `\s*$`, in Python's whitespace set rather than JavaScript's.
  // Only the PEP 440 attempt strips: LegacyVersion keeps the string it was handed.
  const m = PEP440.exec(pyStrip(version));
  if (!m) return null;
  const g = m.groups!;

  const release = g.release!.split(".").map((p) => pyInt(p));
  // Trailing zeros are dropped so 1.0 and 1.0.0 compare equal, as packaging's _cmpkey does.
  let end = release.length;
  while (end > 0 && release[end - 1] === 0n) end--;

  const pre = letterVersion(g.pre_l, g.pre_n);
  const post = letterVersion(g.post_l, g.post_n1 ?? g.post_n2);
  const dev = letterVersion(g.dev_l, g.dev_n);
  const pair = (v: { letter: string; n: bigint }): Marker => ({ kind: "pair", ...v });

  return {
    legacy: false,
    epoch: g.epoch ? pyInt(g.epoch) : 0n,
    release: release.slice(0, end),
    // A dev release with no pre and no post is pushed below every pre-release, which is
    // what puts 1.0.dev0 before 1.0a0; otherwise no pre sorts above one.
    pre: pre ? pair(pre) : post === null && dev !== null ? NEG_INF : POS_INF,
    post: post ? pair(post) : NEG_INF,
    dev: dev ? pair(dev) : POS_INF,
    local: parseLocal(g.local),
  };
}

// ---------------------------------------------------------------------------
// The legacy fallback
// ---------------------------------------------------------------------------

/**
 * `_legacy_version_component_re`, `re.compile(r"(\d+ | [a-z]+ | \.| -)", re.VERBOSE)`.
 *
 * `\d` is `\p{Nd}` and not `[0-9]`: a Python str pattern's `\d` is the whole Nd category, so
 * `a٥5` is the two components `a` and `٥5` there and would be the three components
 * `a`, `٥` and `5` under JavaScript's ASCII-only `\d` -- and the third of those is a
 * numeric run, which gets zero-padded and sorts against other numbers. `[a-z]` stays ASCII
 * because the Python pattern carries no IGNORECASE and `str.lower()` has already run.
 */
const LEGACY_COMPONENT = /(\p{Nd}+|[a-z]+|\.|-)/u;

/** `_legacy_version_replacement_map`, as a Map so an inherited key cannot answer a lookup. */
const LEGACY_REPLACEMENTS = new Map<string, string>([
  ["pre", "c"],
  ["preview", "c"],
  ["-", "final-"],
  ["rc", "c"],
  ["dev", "@"],
]);

/**
 * `_parse_version_parts`: numeric runs zero-padded to sort, everything else starred.
 *
 * The padding test is `part[:1] in "0123456789"`, which is ASCII even though the split that
 * produced the part is not -- so a run of Arabic-Indic digits is starred, not padded.
 *
 * The pad itself is `pyZfill`, not `padStart`: `zfill` counts code points and `padStart` counts
 * UTF-16 units, and a numeric run can carry an astral digit -- `\u{1D7D8}` is Nd, so the split
 * above keeps it inside the run. See `pyZfill` for what that costs.
 */
function* legacyParts(s: string): Generator<string> {
  for (const raw of s.split(LEGACY_COMPONENT)) {
    const part = LEGACY_REPLACEMENTS.get(raw) ?? raw;
    if (!part || part === ".") continue;
    yield part[0]! >= "0" && part[0]! <= "9" ? pyZfill(part, 8) : `*${part}`;
  }
  // Alpha / beta / candidate have to sort before final.
  yield "*final";
}

/** `_legacy_cmpkey`, minus the hard-coded epoch of -1 which the key type carries. */
function legacyKey(version: string): string[] {
  const parts: string[] = [];
  for (const part of legacyParts(version.toLowerCase())) {
    if (part.startsWith("*")) {
      // Drop the "-" in front of a prerelease tag ...
      if (comparePyStrings(part, "*final") < 0) {
        while (parts.length > 0 && parts[parts.length - 1] === "*final-") parts.pop();
      }
      // ... and the trailing zeros of each run of numeric parts.
      while (parts.length > 0 && parts[parts.length - 1] === "00000000") parts.pop();
    }
    parts.push(part);
  }
  return parts;
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

/** Galaxy's `parse_version`: PEP 440 where it parses, the setuptools-era key where it does not. */
export function parseVersion(version: string): VersionKey {
  // The legacy epoch is Python's hard-coded -1, which is what puts every unparseable version
  // below every PEP 440 one; a PEP 440 epoch is never negative.
  return pep440Key(version) ?? { legacy: true, epoch: -1n, parts: legacyKey(version) };
}

/**
 * Numbers and bigints only. Strings go through `comparePyStrings`, because Python orders them
 * by code point and JavaScript by UTF-16 code unit -- and the strings compared below are the
 * arbitrary slices of a tag name a legacy key is made of, plus the pre / post / dev letter and
 * the alphanumeric local segments, any of which can carry an astral character.
 */
const cmp = <T extends number | bigint>(a: T, b: T): number => (a < b ? -1 : a > b ? 1 : 0);

/** Python tuple comparison: element by element, then the shorter tuple first. */
function cmpSeq<T>(a: readonly T[], b: readonly T[], each: (x: T, y: T) => number): number {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const c = each(a[i]!, b[i]!);
    if (c !== 0) return c;
  }
  return cmp(a.length, b.length);
}

const RANK: Record<Marker["kind"], number> = { "-inf": -1, pair: 0, inf: 1 };

function cmpMarker(a: Marker, b: Marker): number {
  if (a.kind !== b.kind) return cmp(RANK[a.kind], RANK[b.kind]);
  if (a.kind !== "pair" || b.kind !== "pair") return 0;
  return comparePyStrings(a.letter, b.letter) || cmp(a.n, b.n);
}

function cmpLocal(a: readonly LocalPart[] | null, b: readonly LocalPart[] | null): number {
  // No local segment is NegativeInfinity, so it sorts below any local segment at all.
  if (a === null || b === null) return cmp(a === null ? 0 : 1, b === null ? 0 : 1);
  return cmpSeq(a, b, (x, y) => {
    // The numeric slot holds NegativeInfinity for a string part, so numbers outrank strings.
    if (x.num === null || y.num === null) {
      const c = cmp(x.num === null ? 0 : 1, y.num === null ? 0 : 1);
      if (c !== 0) return c;
    } else if (x.num !== y.num) {
      return cmp(x.num, y.num);
    }
    return comparePyStrings(x.str, y.str);
  });
}

/** Compare two parsed keys the way Python compares `_BaseVersion._key` tuples. */
export function compareVersionKeys(a: VersionKey, b: VersionKey): number {
  if (a.epoch !== b.epoch) return cmp(a.epoch, b.epoch);
  // Equal epochs means both keys are the same kind: only a legacy key is ever negative.
  if (a.legacy || b.legacy) {
    return a.legacy && b.legacy ? cmpSeq(a.parts, b.parts, comparePyStrings) : 0;
  }
  return (
    cmpSeq(a.release, b.release, cmp) ||
    cmpMarker(a.pre, b.pre) ||
    cmpMarker(a.post, b.post) ||
    cmpMarker(a.dev, b.dev) ||
    cmpLocal(a.local, b.local)
  );
}

/** Order two version strings ascending, as `parse_version(a) < parse_version(b)` does. */
export function compareVersions(a: string, b: string): number {
  return compareVersionKeys(parseVersion(a), parseVersion(b));
}
