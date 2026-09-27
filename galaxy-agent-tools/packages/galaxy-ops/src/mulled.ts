/**
 * Resolve the best quay.io/biocontainers image for a set of conda packages.
 *
 * A port of Galaxy's `galaxy.tool_util.deps.mulled.recommend` (and the parts of
 * `mulled.util` it leans on), so the TypeScript surface can answer the same question as
 * the Python tool without galaxy-tool-util. Everything here is deliberately free of the
 * Galaxy client: quay.io is a different host, reached with `globalThis.fetch` the way
 * `iwc-manifest.ts` reaches the IWC manifest.
 *
 * The behaviour it mirrors, which the tests pin case for case:
 *  - one package  -> repo is the package name, tag matched by version, newest as fallback
 *  - several      -> mulled-v2 repo = sha1 of the sorted names, tag = sha1 of the versions
 *  - unpinned in a multi-package set -> try recent versions until a built combination is found
 *  - a missing repository or a transient network failure is a NONE result, not a throw
 *
 * Python reaches quay.io through `requests`, so its error handling is written in terms of
 * that library's exception tree; the two classes below stand in for it so each `except`
 * clause ports across as itself rather than as a guess -- see `QuayRequestError`. The URL it
 * sends is prepared the same way too -- see `quayRepositoryUrl`, because `fetch`'s own URL
 * parsing is not `requests`' quoting.
 *
 * ACCEPTED DIVERGENCES
 *
 * Everything else here is held to the interpreter case by case. These are the inputs where this
 * answers differently on purpose, because matching would mean carrying Python's codec registry,
 * `charset_normalizer`, a pinned copy of CPython's Unicode tables, its JSON decoder, or one
 * runtime's HTTP internals. Each names the input, what each side answers, and what closing it
 * would take. The reference is CPython 3.12.9 (Unicode 15.0) with `requests` 2.34.2, against
 * Node 22.23 (ICU 78.2, Unicode 17.0).
 *
 *  1. A `charset` naming a codec the Encoding Standard does not have. `charset=utf-7` over
 *     `{"tags":{"+ADE-.17":{}}}` is the tag `1.17` in Python -- so `samtools=1.17` is an exact
 *     match -- and the tag `+ADE-.17` here, a name-only fallback, because the decode fails and
 *     `Response.text`'s replacing fallback leaves the bytes as they were. Python has a codec
 *     registry of about a hundred codecs behind it; `TextDecoder` has the Standard's 40, and
 *     utf-7 is in neither that list nor anything portable.
 *  2. A codec both have and spell differently. `charset=ascii` over a tag holding byte 0xE9 is
 *     `1.0--caf\uFFFD` in Python and `1.0--café` here, because the Standard's `ascii` label IS
 *     windows-1252. `pyDecode` undoes the three that matter -- latin-1, the BOM rules and
 *     UTF-32 -- and the rest of that table is not ported.
 *  3. No `Content-Type` at all, over a body no UTF explains. For
 *     `b'{"tags":{"1.0--caf\xe9":{}}}'` `requests` asks `charset_normalizer`, which answers
 *     windows-1250 and reads the tag as `1.0--café`; this reads utf-8 with replacement and gets
 *     `1.0--caf\uFFFD`, a different image rather than a failed lookup. A statistical charset
 *     detector is not something to reimplement.
 *  4. A codec name Python's alias table does not hold. `normalizeCodec` drops every
 *     non-alphanumeric character and matches on what is left, so `charset=utf16le` decodes
 *     UTF-16LE bytes here and is a LookupError there -- Python falls back to a replacing utf-8
 *     decode, which cannot parse, and answers a failed lookup where this answers
 *     `exact_version`. Closing it means porting `encodings.aliases`, not one more name.
 *  5. The same registry from two more angles: a byte a codec both have maps differently, and an
 *     alias only Python holds. `charset=windows-1252` over a tag carrying byte 0x81 is
 *     `1.0--caf\uFFFD` in Python, whose cp1252 table leaves 0x81 unmapped, and `1.0--caf\u0081`
 *     here, because the Standard's windows-1252 index maps that byte to U+0081 -- two different
 *     images for one body. `charset=iso-8859-1:1987` over byte 0xE9 is `1.0--café` in Python,
 *     whose alias table spells that latin-1, and `1.0--caf\uFFFD` here, because `TextDecoder`
 *     does not know the label and the replacing utf-8 fallback runs instead. And two tables
 *     both sides have and fill differently, over a tag `1.0--X` whose X is the bytes named:
 *     `charset=shift_jis` over `81 60` is `1.0--\u301C` in Python, whose shift_jis carries the
 *     JIS mapping and reads WAVE DASH, and `1.0--\uFF5E` here, because the Standard's
 *     shift_jis index is the Microsoft one and reads FULLWIDTH TILDE; `charset=big5` over
 *     `A1 45` is `1.0--\u2022` there, BULLET, and `1.0--\u2027` here, HYPHENATION POINT. Same
 *     class as 1, 2 and 4, and the same answer: the registry, its tables and its aliases, not
 *     one more name.
 *  6. A code point assigned after Unicode 15.0, where Node's tables have an answer and
 *     CPython's do not. Three places it changes a result: `\p{Nd}` in `BUILD_NUMBER`, where
 *     tags `["1.0--a_\u{10D42}", "1.0--b_1"]` put the first newest here and the second there;
 *     `toLowerCase()` in `normalize`, where U+1C89 lowercases to U+1C8A here and stays itself
 *     there, which hashes a different mulled-v2 repository; and `str.isprintable()` in
 *     `pyRepr`, where a name holding U+1FAE9 reads as itself here and as `\U0001fae9` in a
 *     Python note. Closing these means shipping the interpreter's `unicodedata`.
 *  7. The words in `notes` when a 200 will not parse. An empty body is
 *     `lookup failed: Expecting value: line 1 column 1 (char 0)` in Python and
 *     `lookup failed: Unexpected end of JSON input` here; `{"a":}` is
 *     `Expecting value: line 1 column 6 (char 5)` there and `Unexpected token '}', "{"a":}" is
 *     not valid JSON` here. The outcome is the same on both -- a failed lookup, `found: false`,
 *     cached for the same five minutes -- and only the sentence differs, because the decoder
 *     that refused the body is what names the reason. Closing it means porting the positions and
 *     the wording of `json`'s decoder, which is a second parser and not a check.
 *  8. A `%2e` that `requote_uri` decodes back into a dot segment. urllib3 removes the dot
 *     segments before anything is unquoted, so for the repository `a/%2e%2e/b` `requests`
 *     prepares -- and sends -- `/biocontainers/a/../b`; this builds the same string and `fetch`
 *     parses it under the URL Standard, which collapses the segment and asks about
 *     `/biocontainers/b`. Also `%2e` and `%2E` (`/biocontainers/.` there, `/biocontainers/`
 *     here), `%2e%2e` (`/biocontainers/..` there, `/api/v1/repository/` here), `%2e/a` and
 *     `a/%2e/b`. Of 96 names checked against `requests.Request(...).prepare().url` these six are
 *     the only ones that differ, all of them as requests rather than as strings, and nothing
 *     short of bypassing `fetch`'s URL parsing closes them.
 *  9. A connection that opens at once and then says nothing: Python gives up at 12s, this at
 *     24s. `requests` budgets the connect and the first read separately at `timeout` each and
 *     `fetch` cannot see the connect phase, so one timer covers both halves at the most Python
 *     allows between them -- see `fetchWithInactivityTimeout`.
 * 10. A connection that takes between Node's own 10s connect limit and Python's 12s: it fails
 *     here and succeeds there, because that limit is below the budget and out of reach from
 *     here. Both 9 and 10 need a registry that is up but silent for over ten seconds, both end
 *     in a failed lookup rather than a wrong image, and the only hook that would close them is
 *     an undici-specific dispatcher -- one runtime's internals for two cases at the edge of a
 *     twelve-second timeout.
 * 11. Headers that keep arriving and finish late: a header line at 0s, 10s and 20s, with the
 *     headers and the body complete at 30s. Python's 12s is a budget per read -- urllib3 sets
 *     it on the socket once and each `readline` over the header lines gets it whole -- so every
 *     line renews it and `samtools=1.17` resolves there at 30s. The doubled window here is one
 *     deadline: the timer armed before the request is re-armed only once `fetch` resolves,
 *     which is when the headers are complete, so this aborts at 24s and caches the failure.
 *     Third edge of the same design as 9 and 10 -- one timer where `requests` has two phases
 *     and then a budget per read -- and the same undici dispatcher is the only hook.
 * 12. A malformed number literal of more than 4,300 digits, where the two disagree about which
 *     error the body earns. Python's decoder is one left-to-right pass and the conversion
 *     happens inside it: `NUMBER_RE` in `json/scanner.py` is
 *     `(-?(?:0|[1-9][0-9]*))(\.[0-9]+)?([eE][-+]?[0-9]+)?`, and `_scan_once` hands the integer
 *     group to `parse_int` before `JSONObject` in `json/decoder.py` gets back control to look
 *     for a `,` or a `}` -- so `int()`'s digit limit fires during the scan and reports ahead of
 *     the syntax error sitting after it. Both directions follow. `{"tags":{"1.17":N.}}` where N
 *     is 4,301 nines offers nothing for `(\.[0-9]+)` to take, so the integer group is all 4,301
 *     digits and Python raises the uncaught `ValueError` from the limit; here `JSON_NUMBER`
 *     takes the trailing `.` into the literal, which leaves no integer to count, so the body
 *     reaches `JSON.parse`, fails there, and is a failed lookup cached for five minutes (`Ne`
 *     and `Ne+` the same). `{"tags":{"1.17":Z}}` where Z is 4,301 zeroes, with or without a
 *     leading `-`, is one `0` to that alternation and then `Expecting ',' delimiter: line 1
 *     column 18 (char 17)` -- column 19 (char 18) with the sign -- a cached failed lookup in
 *     Python, where `JSON_NUMBER` takes the
 *     whole run as an integer and `pyJsonLoads` refuses it with the uncaught digit-limit error.
 *     Closing it means porting the decoder's scan order, which is where each error is raised,
 *     and not a check.
 * 13. A body nested 10,000 deep around a value. `{"tags":{"1.17--h0_0":` + 10,000 `[` + `0` +
 *     10,000 `]` + `}}` is an uncaught `RecursionError` in Python (`maximum recursion depth
 *     exceeded while decoding a JSON array from a unicode string`) and `samtools:1.17--h0_0`,
 *     `exact_version`, here. The limit that fires is the C scanner's own, not the interpreter's:
 *     `json.loads` installs `_json`, which raises at a depth of 9,996 whether
 *     `sys.setrecursionlimit` is 100 or 100,000, where `sys.getrecursionlimit()` drives only the
 *     pure-Python fallback scanner and stops it at 496 under the default 1,000. V8's parser has
 *     no depth of its own to reach -- two million deep still parses -- so matching the limit
 *     means counting depth inside a parser that is not ours.
 * 14. An empty tag name in a package repository, which is the one item here where the
 *     difference is Galaxy's own rather than a runtime's. For the packages `["a", "b"]`, with
 *     `{"tags":{"abc-0":{}}}` for their mulled-v2 repository and `{"tags":{"":{}}}` for each
 *     package's own, the empty tag becomes a candidate version, `build_target` keeps it as
 *     `""`, and `v2_image_name` writes no version hash when no target has a version, so the
 *     image name that gets hashed has no `:` in it: Galaxy raises `IndexError:
 *     list index out of range` at `mulled/recommend.py:411`, which is
 *     `version_hash = v2_image_name(targets).split(":")[1]`. This skips the combination and
 *     answers a found, name_only
 *     `mulled-v2-fcd127ffa1016069006ad91f3f361248f9bdf272:abc-0`. Galaxy's crash on a
 *     degenerate response is not reproduced as a crash; skipping is what the recommender does
 *     for every other tag whose version hash it cannot read.
 *
 * Four places a difference can come from, said once so the list stops growing by one per
 * reading. Only UTF-8, UTF-16, UTF-32 and latin-1 answer here as Python's codecs answer; every
 * other codec's mapping table, alias and error behaviour is Node's `TextDecoder`. The JSON
 * decoder underneath is V8's -- `pyJsonLoads` puts Python's accept/reject decisions on top of
 * it, but the words, the positions, the depth limit and the order a failed or degenerate parse
 * raises them in are V8's. The HTTP read loop is `fetch`'s, so the timing at the edge of a
 * twelve-second budget is `fetch`'s. And where Galaxy's own recommender raises on a degenerate
 * response, this does not promise the same exception: item 14 skips where Galaxy indexes past
 * the end, while a body whose `tags` is not an object raises on both sides under different
 * names (`AttributeError` there, `QuayMalformedResponseError` here). A difference in one of these
 * four places on an input a real registry does not send is on this list by decision, not by
 * oversight.
 *
 * What is NOT on this list, because the two agree: a malformed UTF-32 body under
 * `errors="replace"`. `{"tags":{"1.0--X":{}}}` in UTF-32LE with X's four bytes replaced by
 * `00 D8 00 00` and `charset=utf-32-le` declared is the tag `1.0--\uFFFD` on both -- invalid
 * UTF-32 can be valid JSON -- and the replacement count is one per four-byte unit plus one for a
 * truncated tail on both, over 46,008 bodies compared against Python's three UTF-32 codecs. An
 * earlier revision of this list claimed the count as a divergence and argued a bad body could
 * not be JSON; both halves of that were wrong, and `decodeUtf32` is why the first one is.
 */
import { createHash } from "node:crypto";
import { GalaxyValidationError } from "./errors";
import { compareVersionKeys, parseVersion, type VersionKey } from "./pep440";
import {
  comparePyStrings,
  isPySpace,
  PY_INT_MAX_STR_DIGITS,
  pyIntFromDigits,
  pyReprList,
  pyStrip,
  pyUtf8EncodeError,
} from "./python-str";

export const BIOCONTAINERS_NAMESPACE = "biocontainers";
export const QUAY_BIOCONTAINERS_PREFIX = "quay.io/biocontainers";
const QUAY_REPOSITORY_API_ORIGIN = "https://quay.io";
const QUAY_REPOSITORY_API_PATH = "/api/v1/repository";
/** Galaxy's MULLED_SOCKET_TIMEOUT, which `requests` reads as a socket inactivity timeout. */
export const QUAY_TIMEOUT_MS = 12_000;
/** Galaxy's RECOMMENDATION_CACHE_EXPIRY, in ms. */
const RECOMMENDATION_CACHE_TTL_MS = 300_000;
const RECOMMENDATION_CACHE_MAXSIZE = 128;
/** Galaxy's MAX_VERSION_CANDIDATES / MAX_VERSION_COMBOS. */
const MAX_VERSION_CANDIDATES = 8;
const MAX_VERSION_COMBOS = 256;

export type RecommendationSource = "quay_single" | "quay_mulled_v2" | "none";
export type MatchQuality = "exact_version" | "name_only" | "not_found";

export interface PackageSpec {
  readonly name: string;
  readonly version?: string | null;
}

export interface ContainerRecommendation {
  readonly image: string | null;
  readonly source: RecommendationSource;
  readonly match_quality: MatchQuality;
  readonly packages: readonly PackageSpec[];
  readonly multi_package: boolean;
  readonly tag: string | null;
  readonly notes: readonly string[];
  readonly found: boolean;
}

/**
 * Stands in for `requests.RequestException`: the request did not come back usable.
 *
 * Python's recommender separates this from an HTTP status because the two mean different
 * things to it -- a status says the registry answered, a request error says it did not --
 * and different `except` clauses catch them. Keeping the same split here is what lets each
 * of those clauses port over literally instead of being re-decided.
 */
export class QuayRequestError extends Error {}

/**
 * Stands in for `requests.HTTPError` (what `raise_for_status` raises), which subclasses
 * `RequestException` there and so subclasses `QuayRequestError` here. Not just a 404: Python's
 * `except HTTPError` covers every 4xx and 5xx, so a repository that answers 500 is "no
 * container" to it exactly as a 404 is. Nothing outside 400-599 is -- raise_for_status is two
 * range tests, so a terminal 3xx and a 600 alike are bodies it lets through.
 */
export class QuayHttpStatusError extends QuayRequestError {}

/**
 * A 200 whose body is not the shape `quay_versions` reads. Deliberately NOT a
 * `QuayRequestError`: Python's is the bare `Exception` it raises for a missing tag list, or
 * the `AttributeError` / `TypeError` a `.keys()` on something that is not a mapping raises,
 * and `recommend_container`'s `except RequestException` catches neither -- a registry
 * answering nonsense is a bug to surface, not a lookup to degrade.
 */
export class QuayMalformedResponseError extends Error {}

function noRecommendation(
  packages: readonly PackageSpec[],
  multiPackage: boolean,
  note: string,
): ContainerRecommendation {
  return {
    image: null,
    source: "none",
    match_quality: "not_found",
    packages,
    multi_package: multiPackage,
    tag: null,
    notes: [note],
    found: false,
  };
}

// --- conda targets ---------------------------------------------------------

/**
 * Galaxy's `SHELL_UNSAFE_PATTERN`, `re.compile(r"[\s\"']")`, over `isPySpace` for the same
 * reason: a JavaScript `\s` class is neither a superset nor a subset of Python's, so it would
 * let U+0085 through and refuse a U+FEFF that Python accepts.
 */
function isShellUnsafe(value: string): boolean {
  for (let i = 0; i < value.length; i += 1) {
    const code = value.charCodeAt(i);
    // 0x22 is a double quote, 0x27 a single one.
    if (isPySpace(code) || code === 0x22 || code === 0x27) return true;
  }
  return false;
}

/**
 * Galaxy's `CondaTarget`: a package name, an optional version and an optional conda build
 * string. Separate from `PackageSpec` because it is a different type in Galaxy too --
 * `recommend.PackageSpec` is the request, `CondaTarget` is what `build_target` turns each
 * entry into before anything is looked up.
 */
export interface CondaTarget {
  readonly package: string;
  readonly version: string | null;
  readonly build: string | null;
}

/**
 * Galaxy's `build_target`, including the validation `CondaTarget.__init__` does.
 *
 * Nothing shell-unsafe -- whitespace, a single quote or a double quote -- may reach a
 * package name, a version or a build string, and a package name may not be empty. These are
 * the conditions and the messages from `conda_util.py`, and like Python's they run before any
 * request goes out: `_recommend_single` and the all-pinned `_recommend_multi` build their
 * targets first, so `samtools=1.17 2` is refused rather than quietly resolved to a newest-tag
 * fallback. (Python's `channel` rule is not here because `build_target` never sets one.)
 *
 * The name is validated as given and then lowercased, because conda package and quay image
 * names are lowercase and `CondaTarget` stores it that way -- so the hash of a set of names
 * does not depend on how the caller capitalised them.
 */
export function buildTarget(
  packageName: string,
  version?: string | null,
  build?: string | null,
): CondaTarget {
  if (isShellUnsafe(packageName) || !packageName) {
    throw new GalaxyValidationError(`Invalid package [${packageName}] encountered.`);
  }
  if (version && isShellUnsafe(version)) {
    throw new GalaxyValidationError(`Invalid version [${version}] encountered.`);
  }
  if (build != null && isShellUnsafe(build)) {
    throw new GalaxyValidationError(`Invalid build [${build}] encountered.`);
  }
  return { package: packageName.toLowerCase(), version: version ?? null, build: build ?? null };
}

// --- hashing ---------------------------------------------------------------

/**
 * `hashlib.sha1(buffer.encode()).hexdigest()`, refusal included.
 *
 * Python encodes the buffer first, and `str.encode()` will not encode a surrogate code point --
 * so a package name or version carrying one raises UnicodeEncodeError before a byte is hashed.
 * That is a ValueError, which `recommend_container`'s `except RequestException` does not catch,
 * so it leaves the recommender and `recommend_biocontainer` re-raises it as a ValueError of its
 * own; `GalaxyValidationError` is this surface's spelling of that, as it is for `build_target`'s
 * refusals. Without the check, Node substitutes U+FFFD and hashes a different set of names,
 * which asks quay.io about a different repository and answers confidently.
 */
const sha1 = (s: string) => {
  const refusal = pyUtf8EncodeError(s);
  if (refusal !== null) throw new GalaxyValidationError(refusal);
  return createHash("sha1").update(s, "utf8").digest("hex");
};

/**
 * Galaxy's `v2_image_name`. One target is `name[:version[--build]]` with no hashing at all;
 * several are `mulled-v2-<sha1 of the names>` with `:<sha1 of the versions>` when any target
 * carries one. Both hashes run over the targets sorted by package name and joined with "\n",
 * an unpinned package contributes the literal "null" to the version buffer, and build strings
 * never enter either hash.
 *
 * Galaxy's eight doctests are reproduced in the test file. The `image_build` parameter is not
 * here because the recommender never passes one -- and the two doctests that do pass one pass
 * `"0"`, which `_simple_image_name` special-cases back to no suffix at all, so those two
 * expect exactly the strings this produces.
 *
 * The sort is `comparePyStrings`, because Python's `sorted(targets, key=...)` orders names by
 * code point and JavaScript's `<` orders them by UTF-16 code unit. The two disagree whenever
 * one name has an astral character and another has a BMP one above U+DFFF, and the name order
 * is what goes into both hashes -- so a set of names the two languages order differently asks
 * quay.io about a different repository. Both sorts are stable, so equal names keep their
 * argument order in both.
 */
export function v2ImageName(targets: readonly CondaTarget[]): string {
  const only = targets.length === 1 ? targets[0] : undefined;
  if (only) {
    if (only.version === null) return only.package;
    const build = only.build === null ? "" : `--${only.build}`;
    return `${only.package}:${only.version}${build}`;
  }
  const ordered = [...targets].sort((a, b) => comparePyStrings(a.package, b.package));
  const packageHash = sha1(ordered.map((t) => t.package).join("\n"));
  const anyVersion = ordered.some((t) => Boolean(t.version));
  const versionHash = anyVersion ? sha1(ordered.map((t) => t.version || "null").join("\n")) : "";
  return versionHash ? `mulled-v2-${packageHash}:${versionHash}` : `mulled-v2-${packageHash}`;
}

/** Split `<repo>:<versionHash>` the way Galaxy's `base_image_name.split(":", 1)` does. */
function splitImageName(image: string): { repo: string; versionHash: string | undefined } {
  const at = image.indexOf(":");
  return at === -1
    ? { repo: image, versionHash: undefined }
    : { repo: image.slice(0, at), versionHash: image.slice(at + 1) };
}

/** The mulled-v2 repo name (names only, so it never carries a version hash). */
export const v2RepoName = (names: readonly string[]): string =>
  splitImageName(v2ImageName(names.map((name) => buildTarget(name)))).repo;

/** The mulled-v2 version hash for a set, or undefined when nothing in it is pinned. */
export const v2VersionHash = (specs: readonly PackageSpec[]): string | undefined =>
  splitImageName(v2ImageName(specs.map((s) => buildTarget(s.name, s.version)))).versionHash;

// --- tag ordering ----------------------------------------------------------

/** Galaxy's `split_tag(tag)[0]`: everything before the LAST "--" is the conda version. */
export function splitTag(tag: string): string {
  const at = tag.lastIndexOf("--");
  return at === -1 ? tag : tag.slice(0, at);
}

/**
 * Galaxy's BUILD_NUMBER_REGEX, `re.compile(r"\d+$")`, searched (not matched) against the tag's
 * version part. Two Python regex semantics have to be spelled out to get it:
 *
 *  - `\d` is the whole Nd category, so `1.0--a_٢` carries build number 2 where a JavaScript
 *    `\d` finds no build number at all and the tag sorts below `1.0--b_1`.
 *  - `$` matches at the end of the string OR just before a single trailing newline, which a
 *    JavaScript `$` without the `m` flag does not. A tag name is a JSON object key and can
 *    carry a newline, and `1.0--a_2\n` has build number 2 in Python. The lookahead is that
 *    rule rather than `\n?$` inside the match, so `group(0)` stays the digits alone.
 */
const BUILD_NUMBER = /\p{Nd}+(?=\n?$)/u;

interface ParsedTag {
  tag: string;
  version: VersionKey;
  buildString: VersionKey;
  /** A bigint for the same reason the version key holds them: Python compares ints exactly. */
  buildNumber: bigint;
}

/** Galaxy's `parse_tag`, decomposing a tag into version, build string and build number. */
function parseTag(tag: string): ParsedTag {
  let version = tag.includes(":") ? tag.slice(tag.lastIndexOf(":") + 1) : tag;
  let buildString = "-1";
  let buildNumber = -1n;
  const m = BUILD_NUMBER.exec(version);
  // `int()`, not `BigInt()`: the match can hold Nd digits outside `[0-9]`, which `BigInt` throws
  // on and `int` reads by each code point's decimal value.
  if (m) buildNumber = pyIntFromDigits(m[0]);
  if (version.includes("--")) {
    const at = version.lastIndexOf("--");
    [version, buildString] = [version.slice(0, at), version.slice(at + 2)];
  } else if (version.includes("-")) {
    // A mulled multi-package tag: <version_hash>-<build>.
    const at = version.lastIndexOf("-");
    [version, buildString] = [version.slice(0, at), version.slice(at + 1)];
  } else {
    // No build string at all. Galaxy resets the number here too, so any tag that does
    // carry a build sorts above this one for the same version.
    buildNumber = -1n;
  }
  // Galaxy parses both halves here, once per tag, and sorts on the parsed values.
  return {
    tag,
    version: parseVersion(version),
    buildString: parseVersion(buildString),
    buildNumber,
  };
}

/** Descending, without the subtraction a pair of bigints does not allow. */
const cmpBuildNumber = (x: bigint, y: bigint) => (x === y ? 0 : x > y ? -1 : 1);

/**
 * Galaxy's `version_sorted`: newest first. Three stable passes, each descending, by build
 * string then build number then version -- which, because the last pass wins, makes the
 * effective key (version, buildNumber, buildString) descending.
 */
export function versionSorted(tags: readonly string[]): string[] {
  const stable = (arr: readonly ParsedTag[], cmp: (x: ParsedTag, y: ParsedTag) => number) =>
    arr
      .map((v, i) => [v, i] as const)
      .sort((p, q) => cmp(p[0], q[0]) || p[1] - q[1])
      .map(([v]) => v);
  let out = stable(tags.map(parseTag), (x, y) => compareVersionKeys(y.buildString, x.buildString));
  out = stable(out, (x, y) => cmpBuildNumber(x.buildNumber, y.buildNumber));
  out = stable(out, (x, y) => compareVersionKeys(y.version, x.version));
  return out.map((p) => p.tag);
}

// --- quay.io ---------------------------------------------------------------

/**
 * What came back: the status, the content type and the body as BYTES.
 *
 * Bytes, not text, because the charset is not this layer's to decide. `requests` decodes a body
 * by what the `Content-Type` says and, failing that, by sniffing it -- so a response declaring
 * `charset=utf-16le` is a tag list in Python and was an unparseable body here, cached as a failed
 * lookup for five minutes. `requestsResponseJson` makes that decision over these bytes.
 */
export interface QuayResponse {
  readonly status: number;
  readonly contentType: string | null;
  readonly body: Uint8Array;
}

/**
 * `requests`' `timeout=` is not a deadline on the exchange: it bounds the connect, and then it
 * bounds each individual read. A large but steadily arriving body still succeeds long after it,
 * where a single `AbortSignal.timeout` would abort -- and the abort is worse than slow, because
 * the failed lookup is then cached for five minutes. So: one timer, armed before the request
 * and re-armed on every chunk of the body.
 *
 * The budget until the headers arrive is TWO timeouts, not one. `requests` spends up to
 * `timeout` establishing the connection and then a fresh `timeout` waiting for the first read
 * to complete, so 24s can legitimately pass before Python sees a status line -- 7s of connect
 * followed by 7s of silence is a healthy response there and was an aborted, cached failure
 * here. `fetch` cannot observe the connect phase separately, so one timer covers both halves
 * at the most Python permits between them, and the per-chunk timer below is the read budget
 * proper. A server that goes quiet for over 24s before its headers are complete is a failed
 * lookup in both; one that keeps sending header lines can take longer than that and still
 * answer in Python, whose budget is per read rather than per exchange -- divergence 11.
 *
 * Python could raise either ConnectTimeout or ReadTimeout inside that first window; the
 * message is `requests`' read wording throughout, because the read budget is the one this
 * timer stands in for.
 *
 * The three boundaries that doubled window leaves are accepted divergences 9, 10 and 11 in the
 * module header. Either way the lookup fails, and `recommendContainer` caches the failure for
 * five minutes, so the two surfaces can disagree about one package for that long -- the direction
 * that matters, turning a healthy response into a cached failure, is the one the doubled budget
 * removes.
 *
 * Exported so the tests can hold it to that semantics with a timeout small enough to run in
 * milliseconds; `quayTagsFor` is the only caller and always passes `QUAY_TIMEOUT_MS`.
 */
export async function fetchWithInactivityTimeout(
  url: string,
  timeoutMs: number,
): Promise<QuayResponse> {
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  let idleOut!: (e: Error) => void;
  const idle = new Promise<never>((_resolve, reject) => {
    idleOut = reject;
  });
  // Every await below races `idle`, which attaches a handler -- but if the timer somehow
  // fires with nothing racing, this keeps it from surfacing as an unhandled rejection.
  idle.catch(() => {});
  const arm = (budgetMs: number) => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      const e = new QuayRequestError(
        `Read timed out. (read timeout=${timeoutMs / 1000}) [${url}]`,
      );
      controller.abort(e);
      idleOut(e);
    }, budgetMs);
  };

  try {
    // Connect plus the first read, which requests budgets separately at timeoutMs each.
    arm(timeoutMs * 2);
    const resp = await Promise.race([globalThis.fetch(url, { signal: controller.signal }), idle]);
    arm(timeoutMs);
    // The body is read whatever the status says, because `requests` reads it first: a session
    // without stream=True touches `r.content` inside send(), and only the caller's
    // raise_for_status() afterwards looks at the status. So a 404 whose body is truncated is a
    // ChunkedEncodingError -- a RequestException, a failed lookup -- rather than the missing
    // repository the status alone would suggest. Reading it here keeps that order.
    //
    // A 204 and the like have no body to stream; the headers were the only activity there is.
    const contentType = resp.headers.get("content-type");
    if (!resp.body) {
      const buffer = await Promise.race([resp.arrayBuffer(), idle]);
      return { status: resp.status, contentType, body: new Uint8Array(buffer) };
    }

    const reader = resp.body.getReader();
    const chunks: Uint8Array[] = [];
    let length = 0;
    for (;;) {
      const { done, value } = await Promise.race([reader.read(), idle]);
      if (done) break;
      arm(timeoutMs);
      chunks.push(value);
      length += value.length;
    }
    const body = new Uint8Array(length);
    let at = 0;
    for (const chunk of chunks) {
      body.set(chunk, at);
      at += chunk.length;
    }
    return { status: resp.status, contentType, body };
  } finally {
    clearTimeout(timer);
  }
}

// --- the response body, as requests decodes it -----------------------------

/**
 * Python's `str.strip(chars)`: any of `chars` from either end, not the substring.
 *
 * `requests`' content-type parser strips `"`, `'` and space from each parameter's key and value,
 * and its space is the ASCII one, not `str.strip()`'s whitespace set -- hence the explicit
 * character list rather than `pyStrip`.
 */
function pyStripChars(value: string, chars: string): string {
  let start = 0;
  let end = value.length;
  while (start < end && chars.includes(value[start]!)) start += 1;
  while (end > start && chars.includes(value[end - 1]!)) end -= 1;
  return value.slice(start, end);
}

/**
 * `requests.utils.get_encoding_from_headers` (2.34.2), over the one header it reads.
 *
 * The name it returns is a Python codec name, or the empty string for a `charset=` with nothing
 * after it -- which `requests` treats as "no encoding" everywhere it is consulted, so it is
 * returned as itself rather than folded into null. Three details are easy to get wrong and all
 * three change what a body decodes as:
 *
 *  - an explicit `charset` wins outright, whatever the media type says;
 *  - `application/json` with no charset is utf-8, and anything with `text` in its media type is
 *    ISO-8859-1 -- HTTP's default, not JSON's;
 *  - the media-type tests are `in` on the string as sent, and 2.34.2 stopped lower-casing it, so
 *    `TEXT/PLAIN` matches neither and the answer is null. (Older `requests` lower-cased it and
 *    would say ISO-8859-1. Ours is the installed version's behaviour.)
 *
 * A repeated `charset` resolves to the last one, because `requests` collects the parameters into
 * a dict before it looks for the key.
 */
export function encodingFromContentType(contentType: string | null | undefined): string | null {
  if (!contentType) return null;
  const tokens = contentType.split(";");
  const mediaType = pyStrip(tokens[0]!);
  const STRIP = "\"' ";
  let charset: string | null = null;
  for (const token of tokens.slice(1)) {
    const param = pyStrip(token);
    const at = param.indexOf("=");
    // A parameter with no `=` is skipped entirely, so `;charset` alone is not a charset.
    if (!param || at === -1) continue;
    if (pyStripChars(param.slice(0, at), STRIP).toLowerCase() === "charset") {
      charset = pyStripChars(param.slice(at + 1), STRIP);
    }
  }
  if (charset !== null) return charset;
  if (mediaType.includes("text")) return "ISO-8859-1";
  if (mediaType.includes("application/json")) return "utf-8";
  return null;
}

/**
 * `requests.utils.guess_json_utf`: which UTF a body with no declared charset is in, by its BOM
 * or by where its null bytes fall. JSON's first two characters are ASCII, which is what makes
 * the null pattern decisive.
 */
function guessJsonUtf(body: Uint8Array): string | null {
  const sample = body.subarray(0, 4);
  const at = (i: number) => sample[i];
  // `sample in (BOM_UTF32_LE, BOM_UTF32_BE)` is equality on the whole four-byte slice.
  const whole = (b0: number, b1: number, b2: number, b3: number) =>
    sample.length === 4 && at(0) === b0 && at(1) === b1 && at(2) === b2 && at(3) === b3;
  if (whole(0xff, 0xfe, 0x00, 0x00) || whole(0x00, 0x00, 0xfe, 0xff)) return "utf-32";
  if (at(0) === 0xef && at(1) === 0xbb && at(2) === 0xbf) return "utf-8-sig";
  if ((at(0) === 0xff && at(1) === 0xfe) || (at(0) === 0xfe && at(1) === 0xff)) return "utf-16";
  let nulls = 0;
  for (const byte of sample) if (byte === 0x00) nulls += 1;
  if (nulls === 0) return "utf-8";
  if (nulls === 2) {
    if (at(0) === 0x00 && at(2) === 0x00) return "utf-16-be";
    if (at(1) === 0x00 && at(3) === 0x00) return "utf-16-le";
  }
  if (nulls === 3) {
    if (at(0) === 0x00 && at(1) === 0x00 && at(2) === 0x00) return "utf-32-be";
    if (at(1) === 0x00 && at(2) === 0x00 && at(3) === 0x00) return "utf-32-le";
  }
  return null;
}

/** `codecs.lookup` raised: the charset names a codec Python does not have. */
class PyLookupError extends Error {}
/** The codec has it but the bytes are not valid in it -- Python's UnicodeDecodeError. */
class PyDecodeError extends Error {}

/**
 * `encodings.normalize_encoding` plus the alias table, far enough to recognise the codecs that
 * can actually be reached: whatever `get_encoding_from_headers` returns from a `charset`, plus
 * `guess_json_utf`'s five answers. Anything else is handed to `TextDecoder` under its own label.
 */
function normalizeCodec(codec: string): string {
  return codec.toLowerCase().replace(/[^a-z0-9]/g, "");
}

/**
 * `bytes.decode(codec, errors)`, for the codecs a quay.io response can name.
 *
 * `TextDecoder` is close to Python's codecs but not the same, and each difference below is one
 * this has to undo rather than inherit:
 *
 *  - `iso-8859-1` is windows-1252 in the Encoding Standard, so 0x80 decodes as U+20AC where
 *    Python's latin-1 gives U+0080. Latin-1 is byte-for-code-point and is done here instead.
 *  - `TextDecoder` swallows a leading BOM; Python's `utf-8` and `utf_16_le` keep it as U+FEFF
 *    and only `utf-8-sig` and `utf_16` remove it. Hence `ignoreBOM` everywhere and an explicit
 *    strip for the two codecs that do remove one.
 *  - there is no UTF-32 in the Encoding Standard at all, and `guess_json_utf` can answer with
 *    three of them, so it is decoded here.
 *
 * `fatal` is the difference between `errors="strict"` and `errors="replace"`, which is exactly
 * the difference between the two callers: `Response.json`'s guess decodes strictly and falls
 * back when that fails, `Response.text` replaces and cannot fail.
 *
 * Accepted divergences 1, 2, 4 and 5 in the module header are all here: the codecs this has no
 * way to reach, the one label the Standard reads differently, `normalizeCodec`'s shortcut, and
 * the bytes and aliases two spellings of the same registry disagree about.
 */
function pyDecode(body: Uint8Array, codec: string, fatal: boolean): string {
  try {
    return decodeWithCodec(body, codec, fatal);
  } catch (e) {
    if (e instanceof PyLookupError || e instanceof PyDecodeError) throw e;
    // `new TextDecoder` refuses a label it does not know, which is Python's LookupError; a
    // `decode` that refuses the bytes under `fatal` throws a TypeError, which is the
    // UnicodeDecodeError -- and `Response.json` has to be able to tell the two apart, because
    // it falls back to `Response.text` on the second and not on the first.
    if (e instanceof RangeError) throw new PyLookupError(`unknown encoding: ${codec}`);
    throw new PyDecodeError(e instanceof Error ? e.message : String(e));
  }
}

function decodeWithCodec(body: Uint8Array, codec: string, fatal: boolean): string {
  const name = normalizeCodec(codec);
  const decode = (label: string, bytes: Uint8Array) =>
    new TextDecoder(label, { fatal, ignoreBOM: true }).decode(bytes);
  const hasBom = (b0: number, b1: number) => body[0] === b0 && body[1] === b1;

  if (name === "utf8" || name === "u8" || name === "utf" || name === "utf8sig") {
    const bom = body[0] === 0xef && body[1] === 0xbb && body[2] === 0xbf;
    return decode("utf-8", name === "utf8sig" && bom ? body.subarray(3) : body);
  }
  if (name === "utf16") {
    // Python's `utf_16` reads the BOM to pick an endianness and removes it; with no BOM it
    // takes the platform's order, which is little-endian everywhere this runs. Only a body
    // that has a BOM reaches here from `guess_json_utf`.
    if (hasBom(0xff, 0xfe)) return decode("utf-16le", body.subarray(2));
    if (hasBom(0xfe, 0xff)) return decode("utf-16be", body.subarray(2));
    return decode("utf-16le", body);
  }
  if (name === "utf16le") return decode("utf-16le", body);
  if (name === "utf16be") return decode("utf-16be", body);
  if (name === "utf32" || name === "utf32le" || name === "utf32be") {
    return decodeUtf32(body, name === "utf32" ? null : name === "utf32le", fatal);
  }
  if (name === "latin1" || name === "latin" || name === "l1" || name === "iso88591") {
    let out = "";
    for (const byte of body) out += String.fromCharCode(byte);
    return out;
  }
  return decode(codec, body);
}

/** Python's `utf_32`, `utf_32_le` and `utf_32_be`, which the Encoding Standard does not have. */
function decodeUtf32(body: Uint8Array, little: boolean | null, fatal: boolean): string {
  let bytes = body;
  let le = little;
  if (le === null) {
    if (body[0] === 0xff && body[1] === 0xfe && body[2] === 0x00 && body[3] === 0x00) {
      [le, bytes] = [true, body.subarray(4)];
    } else if (body[0] === 0x00 && body[1] === 0x00 && body[2] === 0xfe && body[3] === 0xff) {
      [le, bytes] = [false, body.subarray(4)];
    } else {
      le = true;
    }
  }
  const bad = (what: string) => {
    if (fatal) throw new PyDecodeError(`'utf-32' codec can't decode bytes: ${what}`);
    return "�";
  };
  let out = "";
  let i = 0;
  for (; i + 4 <= bytes.length; i += 4) {
    const [a, b, c, d] = [bytes[i]!, bytes[i + 1]!, bytes[i + 2]!, bytes[i + 3]!];
    const cp = le ? a | (b << 8) | (c << 16) | (d << 24) : d | (c << 8) | (b << 16) | (a << 24);
    // Python's UTF-32 codec refuses a surrogate or anything past the last code point.
    if (cp < 0 || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) out += bad("code point out of range");
    else out += String.fromCodePoint(cp);
  }
  if (i !== bytes.length) out += bad("truncated data");
  return out;
}

/**
 * `requests.Response.text`, for the encoding `get_encoding_from_headers` settled on.
 *
 * When that is null, `requests` asks `charset_normalizer` to guess (`apparent_encoding`) and
 * utf-8 with replacement stands in for it here -- which is both what `guess_json_utf` answers
 * for a body with no nulls and what `requests` itself falls back to for an encoding it cannot
 * look up. Accepted divergence 3 in the module header is that guess.
 */
function requestsResponseText(body: Uint8Array, encoding: string | null): string {
  if (body.length === 0) return "";
  try {
    return pyDecode(body, encoding || "utf-8", false);
  } catch (e) {
    // `except (LookupError, TypeError): str(self.content, errors="replace")`.
    if (e instanceof PyLookupError) return pyDecode(body, "utf-8", false);
    throw e;
  }
}

/**
 * `requests.Response.json()`: the decoded text and what `json.loads` made of it.
 *
 * The text comes back with the value because it is the text the parse consumed, and the order of
 * a JSON object's keys only survives in it -- see `readTagNamesInResponseOrder`. The order of
 * operations is `json()`'s: with no usable encoding from the headers, a body over three bytes is
 * sniffed for a UTF and decoded strictly, and only a decode that fails falls through to `text`.
 * A parse that fails on the sniffed text does NOT fall through -- `requests` raises there -- so
 * a JSON error propagates from wherever it happened.
 */
export function requestsResponseJson(
  body: Uint8Array,
  contentType: string | null,
): { value: unknown; text: string } {
  const encoding = encodingFromContentType(contentType);
  if (!encoding && body.length > 3) {
    const guess = guessJsonUtf(body);
    if (guess !== null) {
      let text: string | null = null;
      try {
        text = pyDecode(body, guess, true);
      } catch (e) {
        // Wrong UTF guessed -- usually an 8-bit codec the server did not declare.
        if (!(e instanceof PyDecodeError)) throw e;
      }
      if (text !== null) return { value: pyJsonLoads(text), text };
    }
  }
  const text = requestsResponseText(body, encoding);
  return { value: pyJsonLoads(text), text };
}

/** The values `json.loads` accepts and `JSON.parse` does not. */
const PY_JSON_CONSTANTS = ["NaN", "Infinity", "-Infinity"] as const;
/** A digit run long enough that an integer literal made of it crosses `int()`'s limit. */
const LONG_DIGIT_RUN = new RegExp(`[0-9]{${PY_INT_MAX_STR_DIGITS + 1}}`);
/** A JSON number token, sticky so the walk can read one without slicing the body. */
const JSON_NUMBER = /-?[0-9]*(?:\.[0-9]*)?(?:[eE][-+]?[0-9]*)?/y;

/**
 * `json.loads`, which is Python's JSON dialect and not RFC 8259's.
 *
 * Python's decoder accepts `NaN`, `Infinity` and `-Infinity` as values -- the same three
 * `json.dumps` writes for those floats -- and `JSON.parse` refuses all three. quay.io putting
 * one in a tag's metadata (`{"tags":{"1.17":NaN}}`) is a body Python reads as a tag list and
 * this used to reject, which `quayTagsFor` reports as a failed lookup and `recommendContainer`
 * then caches for five minutes.
 *
 * The three are replaced with `null` rather than parsed into a number, because nothing here
 * reads a value out of the tag map: the tag NAMES come from `readTagNamesInResponseOrder`,
 * which scans the response text and is untouched by this, and the only value the body is read
 * for is a top-level `error_type` string -- which `null` is no more equal to than a float is.
 * What matters is the accept/reject decision, and it is Python's after this.
 *
 * Only where a value may stand, which is the start of the text or just after `[`, `,` or `:`:
 * a bare `NaN` anywhere else is a syntax error in Python too, and stays one here because
 * `null` is no more legal in that position than `NaN` was. A tag NAMED `NaN` is inside a string
 * literal and is skipped, which is what the walk over the text is for.
 *
 * The walk also carries the other direction, a value `JSON.parse` accepts and `json.loads` does
 * not: the decoder reads an integer literal with `int()`, so more than
 * `sys.get_int_max_str_digits()` digits is the same ValueError a build number of that length
 * raises -- `{"tags":{"1.17":<4301 nines>}}` is a ValueError there and the number `Infinity`
 * here, which would have recommended the tag. Only an INTEGER literal: a `.` or an exponent
 * makes it `float()`, which has no limit and answers `inf` in both. The digits are counted
 * without the sign, and JSON has no leading zeros for them to hide behind.
 *
 * Which of the two errors a body gets is Python's as well, and that is what the bracket stack is
 * for. The decoder is one left-to-right pass, so a syntax error BEFORE the long integer is what
 * it reports (`{"a": 1 <4301 nines>}` is `Expecting ',' delimiter`) and one after it is not
 * (`{"a": <4301 nines> x}` is the ValueError, and so is a body that simply stops). `JSON.parse`
 * cannot be asked where its error is without reading its message, so the text up to the integer
 * is closed off with the brackets still open and parsed on its own: if that parses, the integer
 * is the first thing wrong and the ValueError is raised; if it does not, the syntax error came
 * first and `JSON.parse` below reports it, message and all.
 */
function pyJsonLoads(text: string): unknown {
  // The walk only has to run for a body that carries one of the three tokens, or a digit run
  // long enough for the limit to reach it.
  if (!text.includes("NaN") && !text.includes("Infinity") && !LONG_DIGIT_RUN.test(text)) {
    return JSON.parse(text);
  }
  let out = "";
  let i = 0;
  // The start of the text is a value position; `[`, `,` and `:` are the others.
  let valueMayStand = true;
  // The closers of what is still open, so the text before an over-long integer can be parsed.
  const open: string[] = [];
  let tooLong: { prefix: string; closers: string; digits: string } | null = null;
  while (i < text.length) {
    const c = text[i]!;
    if (c === '"') {
      const end = endOfJsonString(text, i);
      // Unterminated: a syntax error either way, and `JSON.parse` is the one that says so.
      if (end < 0) break;
      out += text.slice(i, end);
      i = end;
      valueMayStand = false;
      continue;
    }
    const token = valueMayStand
      ? PY_JSON_CONSTANTS.find((t) => text.startsWith(t, i))
      : undefined;
    if (token !== undefined) {
      out += "null";
      i += token.length;
      valueMayStand = false;
      continue;
    }
    if (c === "-" || (c >= "0" && c <= "9")) {
      JSON_NUMBER.lastIndex = i;
      // Sticky and able to match nothing, so `exec` cannot fail on a character that got here.
      const number = JSON_NUMBER.exec(text)![0];
      const digits = /^-?[0-9]+$/.test(number) ? number.replace("-", "") : "";
      if (tooLong === null && digits.length > PY_INT_MAX_STR_DIGITS) {
        tooLong = { prefix: out, closers: [...open].reverse().join(""), digits };
      }
      out += number;
      i += number.length;
      valueMayStand = false;
      continue;
    }
    // Whitespace between a delimiter and its value does not close the position; `{` is not one
    // of the delimiters, because what follows a `{` is a key.
    if (c === "[" || c === "," || c === ":") valueMayStand = true;
    else if (!isJsonSpaceChar(c)) valueMayStand = false;
    if (c === "{" || c === "[") open.push(c === "{" ? "}" : "]");
    else if (c === "}" || c === "]") open.pop();
    out += c;
    i += 1;
  }
  if (tooLong !== null && parses(`${tooLong.prefix}0${tooLong.closers}`)) {
    // The decoder would have reached the integer, so this is the error it raises there.
    pyIntFromDigits(tooLong.digits);
  }
  return JSON.parse(out + text.slice(i));
}

/** Whether `JSON.parse` accepts a text at all, for the ordering question above. */
function parses(text: string): boolean {
  try {
    JSON.parse(text);
    return true;
  } catch {
    return false;
  }
}

/** A JSON object, and not an array or anything else `.keys()` would refuse in Python. */
function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const proto = Object.getPrototypeOf(value) as unknown;
  return proto === Object.prototype || proto === null;
}

/**
 * The tag names of a body's top-level `tags` object, in the order quay.io sent them, or null
 * when that member is not an object at all.
 *
 * `Object.keys` cannot answer this. JavaScript enumerates an object's integer-shaped keys
 * first, in ascending numeric order, ahead of everything else -- so `{"tags":{"1.0":{},"1":{}}}`
 * comes back as ["1", "1.0"], the two tags tie in `version_sorted`, and an unpinned request
 * picks `:1` where Python picks `:1.0`, because a dict keeps the order `json.loads` inserted.
 * Both are legal conda versions and they are different images. `JSON.parse`'s reviver walks the
 * object it has already built, so it has the same order; the response text is the only place
 * the order survives.
 *
 * So this reads the keys out of the text. It runs on text `JSON.parse` has already accepted,
 * which is what keeps it a scan rather than a second parser: find the last top-level `tags`
 * member -- `json.loads` and `JSON.parse` both keep the last of a repeated key -- and read the
 * keys of the object it holds, stepping over each value without interpreting it. A key that
 * appears twice collapses onto its first position, which is where a Python dict keeps a key
 * that is assigned twice.
 */
export function readTagNamesInResponseOrder(text: string): string[] | null {
  let tagsAt = -1;
  const found = eachMember(text, skipJsonSpace(text, 0), (key, valueAt) => {
    if (key === "tags") tagsAt = valueAt;
  });
  if (!found || tagsAt < 0 || text[tagsAt] !== "{") return null;
  const names: string[] = [];
  const seen = new Set<string>();
  const read = eachMember(text, tagsAt, (key) => {
    if (seen.has(key)) return;
    seen.add(key);
    names.push(key);
  });
  return read ? names : null;
}

/** JSON's four whitespace characters, which is all `JSON.parse` allows between tokens. */
function isJsonSpaceChar(c: string): boolean {
  return c === " " || c === "\t" || c === "\n" || c === "\r";
}

function skipJsonSpace(text: string, i: number): number {
  while (i < text.length && isJsonSpaceChar(text[i]!)) i += 1;
  return i;
}

/** The index just past the string literal that starts at `i`, or -1 if it is unterminated. */
function endOfJsonString(text: string, i: number): number {
  for (i += 1; i < text.length; i += 1) {
    const c = text[i];
    if (c === "\\") i += 1;
    else if (c === '"') return i + 1;
  }
  return -1;
}

/** The index just past the value that starts at `i`, or -1 if the text runs out first. */
function endOfJsonValue(text: string, i: number): number {
  const first = text[i];
  if (first === '"') return endOfJsonString(text, i);
  if (first === "{" || first === "[") {
    let depth = 0;
    while (i < text.length) {
      const c = text[i];
      if (c === '"') {
        i = endOfJsonString(text, i);
        if (i < 0) return -1;
        continue;
      }
      if (c === "{" || c === "[") depth += 1;
      else if (c === "}" || c === "]") {
        depth -= 1;
        if (depth === 0) return i + 1;
      }
      i += 1;
    }
    return -1;
  }
  // A number, true, false or null ends at the first character that cannot belong to one.
  while (i < text.length && !",}] \t\n\r".includes(text[i]!)) i += 1;
  return i;
}

/**
 * Walk the members of the object starting at `i` in text order, reporting each key and where
 * its value begins. False means the text is not an object walked to a closing brace, which for
 * text `JSON.parse` accepted means it was not an object at all.
 */
function eachMember(
  text: string,
  i: number,
  onMember: (key: string, valueAt: number) => void,
): boolean {
  if (text[i] !== "{") return false;
  i = skipJsonSpace(text, i + 1);
  if (text[i] === "}") return true;
  for (;;) {
    if (text[i] !== '"') return false;
    const keyEnd = endOfJsonString(text, i);
    if (keyEnd < 0) return false;
    // The key is a JSON string literal, so the built-in decodes its escapes.
    const key = JSON.parse(text.slice(i, keyEnd)) as string;
    i = skipJsonSpace(text, keyEnd);
    if (text[i] !== ":") return false;
    i = skipJsonSpace(text, i + 1);
    onMember(key, i);
    const valueEnd = endOfJsonValue(text, i);
    if (valueEnd < 0) return false;
    i = skipJsonSpace(text, valueEnd);
    if (text[i] === ",") {
      i = skipJsonSpace(text, i + 1);
      continue;
    }
    return text[i] === "}";
  }
}

// --- the URL requests would have sent --------------------------------------

/** RFC 3986's unreserved set: urllib3's `_UNRESERVED_CHARS`, and what `requote_uri` decodes to. */
const URI_UNRESERVED = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-~";
/** urllib3's `_PATH_CHARS`: unreserved, the sub-delimiters and `:@/`. Deliberately no `%`. */
const URI_PATH_CHARS = new Set([...URI_UNRESERVED, ..."!$&'()*+,;=", ...":@/"]);
/** urllib3's `_QUERY_CHARS`, which is the same set as its `_FRAGMENT_CHARS`. */
const URI_QUERY_CHARS = new Set([...URI_PATH_CHARS, "?"]);
const IS_URI_UNRESERVED = /^[A-Za-z0-9._~-]$/;
/** urllib3's `_PERCENT_RE`. */
const PERCENT_ESCAPE = /%[0-9A-Fa-f]{2}/g;

/**
 * `component.encode("utf-8", "surrogatepass")`, the encode urllib3 uses on a URL component.
 *
 * `TextEncoder` substitutes U+FFFD for a lone surrogate; `surrogatepass` encodes it as the
 * three bytes its code point would have, so a repository name carrying one asks quay.io about
 * `%ED%A0%80` in Python and would have asked about `%EF%BF%BD` here. This is the one place
 * Python encodes a surrogate rather than refusing it -- the hashes still refuse, because
 * `str.encode()` there is strict.
 */
function utf8BytesSurrogatepass(value: string): number[] {
  const out: number[] = [];
  for (const ch of value) {
    const cp = ch.codePointAt(0)!;
    if (cp < 0x80) out.push(cp);
    else if (cp < 0x800) out.push(0xc0 | (cp >> 6), 0x80 | (cp & 0x3f));
    else if (cp < 0x10000) {
      out.push(0xe0 | (cp >> 12), 0x80 | ((cp >> 6) & 0x3f), 0x80 | (cp & 0x3f));
    } else {
      out.push(
        0xf0 | (cp >> 18),
        0x80 | ((cp >> 12) & 0x3f),
        0x80 | ((cp >> 6) & 0x3f),
        0x80 | (cp & 0x3f),
      );
    }
  }
  return out;
}

/**
 * urllib3's `_encode_invalid_chars`: percent-encode what the component may not hold, without
 * re-encoding what is already encoded.
 *
 * Every valid `%XX` is uppercased first, and the component's `%` survive only if ALL of them
 * begin one -- otherwise every `%` becomes `%25`, so `a%zzb` is a repository called `a%zzb` and
 * not one called `a` with a mangled escape. The test is per component, which is why the path and
 * the query each make it separately.
 */
function encodeUriChars(component: string, allowed: ReadonlySet<string>): string {
  let escapes = 0;
  const normalized = component.replace(PERCENT_ESCAPE, (m) => {
    escapes += 1;
    return m.toUpperCase();
  });
  let percents = 0;
  for (const ch of normalized) if (ch === "%") percents += 1;
  const percentEncoded = escapes === percents;
  let out = "";
  for (const byte of utf8BytesSurrogatepass(normalized)) {
    const ch = String.fromCharCode(byte);
    if ((percentEncoded && ch === "%") || (byte < 0x80 && allowed.has(ch))) out += ch;
    else out += `%${byte.toString(16).toUpperCase().padStart(2, "0")}`;
  }
  return out;
}

/**
 * urllib3's `_remove_path_dot_segments`: RFC 3986 section 5.2.4 over the path it parsed.
 *
 * Its place in the order is the whole point. urllib3 runs this on the path as written, BEFORE
 * `_encode_invalid_chars` and long before `requests` unquotes anything, so a `%2e` is not a dot
 * segment when this runs and a literal `..` is. `a/%2e/../b` therefore pops the `%2e` and asks
 * about `a/b`; doing it the other way round pops the `a` and asks about `b`, which is a
 * different repository. It also runs over the path of the whole URL, not the repository name, so
 * enough `..` climbs out of `/api/v1/repository/biocontainers/` -- `../a` is a request to
 * `/api/v1/repository/a`.
 *
 * The two trailing rules are urllib3's: a path that started with `/` keeps one, and a path
 * ending in `/.` or `/..` keeps a trailing slash.
 */
function removePathDotSegments(path: string): string {
  const output: string[] = [];
  for (const segment of path.split("/")) {
    if (segment === ".") continue;
    if (segment !== "..") output.push(segment);
    else if (output.length > 0) output.pop();
  }
  if (path.startsWith("/") && (output.length === 0 || output[0])) output.unshift("");
  if (path.endsWith("/.") || path.endsWith("/..")) output.push("");
  return output.join("/");
}

/**
 * `requests.PreparedRequest.prepare_url` for the one URL this module builds.
 *
 * Interpolating the repository name raw and handing the string to `fetch` asks a different
 * question than Python asks. `fetch` parses what it is given under the URL Standard, which for
 * an https URL turns a backslash into a path separator: `a\b` becomes `/biocontainers/a/b`,
 * a repository that is not the one Python asks about -- `requests` prepares
 * `/biocontainers/a%5Cb`. The Standard also leaves `|`, `^`, `[` and `]` in a path where Python
 * encodes them, and normalises a `%XX` differently.
 *
 * So the preparation is ported instead, which is two steps and not one:
 *
 *  - urllib3's `parse_url` splits the fragment at the first `#` and the query at the first `?`
 *    before anything is encoded -- the prefix here has neither -- then removes the path's dot
 *    segments (see `removePathDotSegments`), and only then percent-encodes each component
 *    against its own allowed set. An empty query or fragment is dropped by `urlunparse` rather
 *    than leaving a bare `?` or `#`.
 *  - `requests`' own `requote_uri` then decodes every escape that stands for an unreserved
 *    character, so a caller's `%41` is an `A` and `%7E` a `~` in the request Python sends. The
 *    other half of `requote_uri` -- `quote` over the result -- cannot change anything after
 *    urllib3 has run, and its `InvalidURL` fallback cannot be reached, because urllib3 leaves no
 *    `%` that does not begin a valid escape.
 *
 * Checked case for case against `requests.Request("GET", url).prepare().url` on the installed
 * 2.34.2 -- backslashes, the unsafe ASCII, interior whitespace, non-ASCII, a lone surrogate,
 * every `%` shape, `.` and `..` segments, and a name carrying `?` or `#` -- through a `new URL()`
 * round trip, so what is compared is the URL `fetch` requests rather than the string handed to it.
 */
function quayRepositoryUrl(repo: string): string {
  const hashAt = repo.indexOf("#");
  const fragment = hashAt === -1 ? "" : repo.slice(hashAt + 1);
  const beforeHash = hashAt === -1 ? repo : repo.slice(0, hashAt);
  const queryAt = beforeHash.indexOf("?");
  const query = queryAt === -1 ? "" : beforeHash.slice(queryAt + 1);
  const name = queryAt === -1 ? beforeHash : beforeHash.slice(0, queryAt);
  const path = removePathDotSegments(
    `${QUAY_REPOSITORY_API_PATH}/${BIOCONTAINERS_NAMESPACE}/${name}`,
  );
  let url = `${QUAY_REPOSITORY_API_ORIGIN}${encodeUriChars(path, URI_PATH_CHARS)}`;
  if (query) url += `?${encodeUriChars(query, URI_QUERY_CHARS)}`;
  if (fragment) url += `#${encodeUriChars(fragment, URI_QUERY_CHARS)}`;
  // `requote_uri`'s `unquote_unreserved`, over the whole URL as `requests` runs it.
  return url.replace(PERCENT_ESCAPE, (m) => {
    const ch = String.fromCharCode(parseInt(m.slice(1), 16));
    return IS_URI_UNRESERVED.test(ch) ? ch : m;
  });
}

/**
 * Galaxy's `quay_versions` + `mulled_tags_for`: one unauthenticated GET, no pagination,
 * "latest" dropped, newest first.
 *
 * Errors keep Python's shape. A 4xx or 5xx is what `raise_for_status` turns into an HTTPError; a
 * transport failure, a read that stalls past the socket timeout, or a body that will not parse
 * is a RequestException (`requests` raises its own `JSONDecodeError`, which subclasses one).
 *
 * A 200 whose body is not the shape Python reads is neither, and propagates: Python raises a
 * bare Exception for a missing `tags` key, and `.keys()` on anything that is not a mapping --
 * a list of tag names, say -- raises an AttributeError. Reading such a body leniently is how
 * `{"tags": ["1.17--h0_0"]}` becomes a confidently verified `samtools:0`.
 */
export async function quayTagsFor(repo: string): Promise<string[]> {
  // Percent-encoded as `requests` would encode it, not as `fetch`'s URL parser would -- see
  // quayRepositoryUrl. The prepared URL is also what goes in the messages below, because that
  // is the URL Python's HTTPError names.
  const url = quayRepositoryUrl(repo);
  let resp: QuayResponse;
  try {
    resp = await fetchWithInactivityTimeout(url, QUAY_TIMEOUT_MS);
  } catch (e) {
    if (e instanceof QuayRequestError) throw e;
    throw new QuayRequestError(e instanceof Error ? e.message : String(e));
  }
  // raise_for_status() rejects 400-599 and nothing else -- it is two range tests, not "not ok"
  // and not ">= 400". A 1xx or a terminal 3xx is a body Python goes on to parse, so a redirect
  // fetch could not follow is not "no container"; and so is a 600, which is outside HTTP's
  // registered range but is a status line a server can send and undici will hand over.
  if (resp.status >= 400 && resp.status < 600) {
    throw new QuayHttpStatusError(`${resp.status} Error for url: ${url}`);
  }
  // `response.json()`, charset and all: `requests` raises its own JSONDecodeError, which
  // subclasses RequestException, so an unparseable body is a failed lookup either way. The one
  // error that is not a JSONDecodeError is `int()`'s over an integer literal too long to read:
  // `Response.json` re-raises only a JSONDecodeError, so that ValueError leaves the recommender
  // the way a build number of that length does rather than degrading to a failed lookup.
  let parsed: { value: unknown; text: string };
  try {
    parsed = requestsResponseJson(resp.body, resp.contentType);
  } catch (e) {
    if (e instanceof GalaxyValidationError) throw e;
    throw new QuayRequestError(e instanceof Error ? e.message : String(e));
  }
  const data = parsed.value;
  // Python's message ends with a repr of the whole body; this names the repository instead,
  // which is the part a reader needs and the part that reprs the same in both languages.
  const malformed = (what: string) =>
    new QuayMalformedResponseError(`Unexpected response from quay.io - ${what} [${url}]`);
  if (!isPlainObject(data)) throw malformed("no tags description found");
  if (data.error_type === "invalid_token") return [];
  if (!Object.hasOwn(data, "tags")) throw malformed("no tags description found");
  // The response order, which Object.keys would lose -- see readTagNamesInResponseOrder. It
  // also answers "is the tag list a mapping": null means that member is something else.
  const names = readTagNamesInResponseOrder(parsed.text);
  if (names === null) throw malformed("the tag list is not a mapping of tag names");
  return versionSorted(names.filter((t) => t !== "latest"));
}

/** Python's `except HTTPError: tags = []` around a tag fetch. */
async function tagsOrEmptyOnStatus(repo: string): Promise<string[]> {
  try {
    return await quayTagsFor(repo);
  } catch (e) {
    if (e instanceof QuayHttpStatusError) return [];
    throw e;
  }
}

// --- tag selection ---------------------------------------------------------

export interface TagChoice {
  tag: string | null;
  exact: boolean;
}

/** Galaxy's `select_single_package_tag` (`tags` newest-first). */
export function selectSinglePackageTag(
  tags: readonly string[],
  version: string | null | undefined,
  opts: { allowNewestFallback?: boolean } = {},
): TagChoice {
  const newest = tags[0];
  if (newest === undefined) return { tag: null, exact: false };
  if (version != null) {
    for (const tag of tags) if (splitTag(tag) === version) return { tag, exact: true };
  }
  return opts.allowNewestFallback ? { tag: newest, exact: false } : { tag: null, exact: false };
}

/**
 * Galaxy's `select_mulled_v2_tag`. Without a hash the newest tag IS the answer and counts as
 * exact -- there is no finer version for it to mismatch.
 */
export function selectMulledV2Tag(
  tags: readonly string[],
  versionHash: string | null | undefined,
  opts: { allowNewestFallback?: boolean } = {},
): TagChoice {
  const newest = tags[0];
  if (newest === undefined) return { tag: null, exact: false };
  if (versionHash) {
    for (const tag of tags) {
      if (tag === versionHash || tag.startsWith(`${versionHash}-`)) return { tag, exact: true };
    }
    return opts.allowNewestFallback ? { tag: newest, exact: false } : { tag: null, exact: false };
  }
  return { tag: newest, exact: true };
}

/**
 * Galaxy's `find_remote_mulled_name` with the recommender's `_lookup_exact_or_newest` error
 * handling: an HTTP status (repo absent, or the registry refusing) is "not found"; a request
 * error propagates to `recommendContainer`'s outer handler as a lookup failure.
 */
async function lookupExactOrNewest(
  targets: readonly CondaTarget[],
): Promise<{ name: string; exact: boolean } | null> {
  const single = targets.length === 1 ? targets[0] : undefined;
  let tags: string[];
  const { repo, versionHash } = single
    ? { repo: single.package, versionHash: undefined }
    : splitImageName(v2ImageName(targets));
  try {
    tags = await quayTagsFor(repo);
  } catch (e) {
    if (e instanceof QuayHttpStatusError) return null;
    throw e;
  }
  const { tag, exact } = single
    ? selectSinglePackageTag(tags, single.version, { allowNewestFallback: true })
    : selectMulledV2Tag(tags, versionHash, { allowNewestFallback: true });
  return tag === null ? null : { name: `${repo}:${tag}`, exact };
}

// --- the TTL cache ---------------------------------------------------------

interface CacheEntry {
  expiresAt: number;
  value: ContainerRecommendation;
}

/**
 * Python's `_TTLCache` reads `time.monotonic()`, which never goes backwards. `Date.now()`
 * does -- an NTP correction or a manual clock change is enough -- and a backwards step
 * extends every live entry by however far the clock moved, so a negative recommendation
 * outlives its five minutes. `performance.now()` is the monotonic clock here.
 */
const monotonicNow = () => performance.now();

const cache = new Map<string, CacheEntry>();

function cacheGet(key: string): ContainerRecommendation | null {
  const entry = cache.get(key);
  if (!entry) return null;
  if (monotonicNow() >= entry.expiresAt) {
    cache.delete(key);
    return null;
  }
  return entry.value;
}

function cacheSet(key: string, value: ContainerRecommendation): void {
  if (!cache.has(key) && cache.size >= RECOMMENDATION_CACHE_MAXSIZE) {
    // Evict the entry closest to expiry, as Galaxy's _TTLCache does.
    let soonest: string | null = null;
    let at = Infinity;
    for (const [k, v] of cache) if (v.expiresAt < at) [soonest, at] = [k, v.expiresAt];
    if (soonest !== null) cache.delete(soonest);
  }
  cache.set(key, { expiresAt: monotonicNow() + RECOMMENDATION_CACHE_TTL_MS, value });
}

/** Tests only: the cache is module-wide and would otherwise leak between cases. */
export function __clearRecommendationCacheForTest(): void {
  cache.clear();
}

/**
 * Galaxy's `(_cache_key(normalized), resolve_versions)`, rendered as a string key.
 *
 * `_cache_key` sorts whole `(name, version)` tuples, so the version breaks a tie between two
 * entries with the same name. Sorting on the name alone leaves `bwa=0.7.17, bwa=0.7.18` and
 * its reverse with different keys -- two lookups for one request, and the second can resolve
 * to a different version hash than the first.
 *
 * Sorted with `comparePyStrings` for the reason `v2ImageName` is: `sorted()` on a tuple of
 * strings is a code-point order, and two spellings of one request that JavaScript orders the
 * other way round would be two cache entries where Python has one.
 */
const cacheKey = (specs: readonly PackageSpec[], resolveVersions: boolean) =>
  JSON.stringify([
    specs
      .map((p) => [p.name, p.version || ""] as const)
      .sort((a, b) => comparePyStrings(a[0], b[0]) || comparePyStrings(a[1], b[1])),
    resolveVersions,
  ]);

// --- the recommendation ----------------------------------------------------

const normalize = (specs: readonly PackageSpec[]): PackageSpec[] =>
  specs
    .filter((p) => p.name)
    .map((p) => ({
      name: pyStrip(p.name).toLowerCase(),
      version: p.version ? pyStrip(p.version) || null : null,
    }));

async function recommendSingle(spec: PackageSpec): Promise<ContainerRecommendation> {
  const match = await lookupExactOrNewest([buildTarget(spec.name, spec.version)]);
  if (match === null) {
    return noRecommendation([spec], false, `no biocontainer found for '${spec.name}'`);
  }
  const tag = match.name.slice(match.name.indexOf(":") + 1);
  const notes =
    spec.version && !match.exact
      ? [`requested version ${spec.version} not found; newest available tag is ${tag}`]
      : [];
  return {
    image: `${QUAY_BIOCONTAINERS_PREFIX}/${match.name}`,
    source: "quay_single",
    match_quality: match.exact ? "exact_version" : "name_only",
    packages: [spec],
    multi_package: false,
    tag,
    notes,
    found: true,
  };
}

/** Galaxy's `_candidate_versions`: recent versions for one package, newest first. */
async function candidateVersions(name: string): Promise<string[]> {
  let tags: string[];
  try {
    tags = await quayTagsFor(name);
  } catch (e) {
    // Python catches (HTTPError, RequestException) here, which is just RequestException.
    if (e instanceof QuayRequestError) return [];
    throw e;
  }
  const versions: string[] = [];
  for (const tag of tags) {
    const v = splitTag(tag);
    if (!versions.includes(v)) versions.push(v);
    if (versions.length >= MAX_VERSION_CANDIDATES) break;
  }
  return versions;
}

/** The cartesian product, all-newest first, as `itertools.product` yields it. */
function* product<T>(lists: readonly (readonly T[])[]): Generator<T[]> {
  if (lists.length === 0 || lists.some((l) => l.length === 0)) return;
  const idx: number[] = new Array<number>(lists.length).fill(0);
  for (;;) {
    yield lists.map((list, i) => list[idx[i]!]!);
    let i = lists.length - 1;
    for (; i >= 0; i--) {
      if (++idx[i]! < lists[i]!.length) break;
      idx[i] = 0;
    }
    if (i < 0) return;
  }
}

/**
 * Galaxy's `_resolve_versions`: the first candidate combination whose image is built.
 *
 * The `{name: version}` table is a `Map`, not an object literal, because the names are the
 * caller's: `resolved["__proto__"] = "1.0"` on a `{}` runs `Object.prototype`'s setter instead
 * of adding a property, and the read back gives `[object Object]`. A package really can be
 * called `__proto__` -- conda would let you publish one -- and Python's dict just holds it, so
 * the note says `__proto__=1.0` there and said `__proto__=[object Object]` here.
 */
async function resolveVersionsFor(
  specs: readonly PackageSpec[],
  existingTags: readonly string[],
): Promise<{ resolved: Map<string, string> | null; versionHash: string | null }> {
  const none = { resolved: null, versionHash: null };
  const existingHashes = new Set(
    existingTags.filter((t) => t.includes("-")).map((t) => t.slice(0, t.lastIndexOf("-"))),
  );
  if (existingHashes.size === 0) return none;

  const candidateLists: string[][] = [];
  for (const spec of specs) {
    candidateLists.push(spec.version ? [spec.version] : await candidateVersions(spec.name));
  }
  if (candidateLists.some((c) => c.length === 0)) return none;

  let count = 0;
  for (const combo of product(candidateLists)) {
    if (count++ >= MAX_VERSION_COMBOS) break;
    const targets = specs.map((spec, i) => buildTarget(spec.name, combo[i]!));
    const versionHash = splitImageName(v2ImageName(targets)).versionHash;
    if (versionHash && existingHashes.has(versionHash)) {
      // Keyed by spec.name, which is what Python's dict comprehension keys on -- not the
      // target's lowercased package, which would be a second pass of a mapping that is not
      // guaranteed to be idempotent.
      const resolved = new Map<string, string>();
      specs.forEach((spec, i) => resolved.set(spec.name, combo[i]!));
      return { resolved, versionHash };
    }
  }
  return none;
}

async function recommendMulti(
  specs: readonly PackageSpec[],
  resolveVersions: boolean,
): Promise<ContainerRecommendation> {
  const names = specs.map((p) => p.name);
  const versioned = specs.filter((p) => p.version);

  // Fully pinned: one probe gives the exact image, or the newest fallback.
  if (versioned.length === specs.length) {
    const match = await lookupExactOrNewest(specs.map((p) => buildTarget(p.name, p.version)));
    if (match === null) {
      // `f"... for {[p.name for p in specs]}"` formats the list with repr, so each name is
      // quoted and escaped the way Python escapes it -- backslashes doubled, non-printables
      // spelled out. Interpolating the names directly changed the note the caller gets back.
      return noRecommendation(
        specs,
        true,
        `no multi-package biocontainer built for ${pyReprList(names)}`,
      );
    }
    return {
      image: `${QUAY_BIOCONTAINERS_PREFIX}/${match.name}`,
      source: "quay_mulled_v2",
      match_quality: match.exact ? "exact_version" : "name_only",
      packages: specs,
      multi_package: true,
      tag: match.name.slice(match.name.indexOf(":") + 1),
      notes: match.exact
        ? []
        : ["exact version combination not built; using newest available mulled-v2 build"],
      found: true,
    };
  }

  // Partial / unpinned: the mulled-v2 repo depends only on the names, so one fetch serves
  // both the tag list and the version resolution below.
  const repo = v2RepoName(names);
  const tags = await tagsOrEmptyOnStatus(repo);
  const newest = tags[0];
  if (newest === undefined) {
    return noRecommendation(
      specs,
      true,
      `no multi-package biocontainer built for ${pyReprList(names)}`,
    );
  }

  const notes: string[] = [];
  let versionHash: string | null = null;
  if (resolveVersions) {
    const out = await resolveVersionsFor(specs, tags);
    versionHash = out.versionHash;
    const resolved = out.resolved;
    if (resolved) {
      const unpinned = specs.filter((p) => !p.version).map((p) => p.name);
      notes.push(
        `resolved version(s): ${unpinned.map((n) => `${n}=${resolved.get(n)!}`).join(", ")}`,
      );
    } else {
      notes.push("could not resolve a built version combination; using newest available build");
    }
  } else if (versioned.length > 0) {
    notes.push("not all packages were versioned; resolved by package names only");
  }

  const tag =
    (versionHash ? selectMulledV2Tag(tags, versionHash, { allowNewestFallback: true }).tag : newest) ??
    newest;
  return {
    image: `${QUAY_BIOCONTAINERS_PREFIX}/${repo}:${tag}`,
    source: "quay_mulled_v2",
    match_quality: versionHash ? "exact_version" : "name_only",
    packages: specs,
    multi_package: true,
    tag,
    notes,
    found: true,
  };
}

/**
 * Resolve the best quay.io/biocontainers image for `packages`.
 *
 * Never throws for a missing container or a transient network failure -- a `none`
 * recommendation with an explanatory note comes back instead, so callers degrade. A quay.io
 * response that is neither, such as a 200 whose body has no tag list, is a bug rather than a
 * degraded lookup and propagates, exactly as Python's narrow `except RequestException` lets
 * it.
 */
export async function recommendContainer(
  packages: readonly PackageSpec[],
  opts: { useCache?: boolean; resolveVersions?: boolean } = {},
): Promise<ContainerRecommendation> {
  const useCache = opts.useCache ?? true;
  const resolveVersions = opts.resolveVersions ?? true;
  const normalized = normalize(packages);
  if (normalized.length === 0) return noRecommendation(normalized, false, "no packages supplied");

  const key = cacheKey(normalized, resolveVersions);
  if (useCache) {
    const hit = cacheGet(key);
    if (hit) return hit;
  }

  const only = normalized.length === 1 ? normalized[0] : undefined;
  let recommendation: ContainerRecommendation;
  try {
    recommendation = only
      ? await recommendSingle(only)
      : await recommendMulti(normalized, resolveVersions);
  } catch (e) {
    if (!(e instanceof QuayRequestError)) throw e;
    recommendation = noRecommendation(normalized, normalized.length > 1, `lookup failed: ${e.message}`);
  }

  if (useCache) cacheSet(key, recommendation);
  return recommendation;
}

/**
 * Tri-state check of whether an image's exact tag is built.
 *
 * true  -- the tag is among the repository's built tags.
 * false -- the repository has tags and this is not one of them, so the reference names a
 *          build that was never published.
 * null  -- cannot be established: not a quay.io/biocontainers reference, no tag, or the tag
 *          list could not be fetched. Deliberately conservative, so a network blip is never
 *          mistaken for a broken reference.
 */
export async function biocontainerTagBuilt(image: string): Promise<boolean | null> {
  if (!image) return null;
  const prefix = `${QUAY_BIOCONTAINERS_PREFIX}/`;
  if (!image.startsWith(prefix)) return null;
  const ref = image.slice(prefix.length);
  if (!ref.includes(":")) return null;
  const at = ref.lastIndexOf(":");
  const repo = ref.slice(0, at);
  const tag = ref.slice(at + 1);
  if (!repo || !tag) return null;

  let tags: string[];
  try {
    tags = await quayTagsFor(repo);
  } catch (e) {
    if (!(e instanceof QuayRequestError)) throw e;
    return null;
  }
  // An empty list cannot tell a missing repo from a blip, so it is not evidence.
  if (tags.length === 0) return null;
  return tags.includes(tag);
}
