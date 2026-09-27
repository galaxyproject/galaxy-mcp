/**
 * The mulled recommender port, held to Galaxy's own unit tests.
 *
 * Every case below is one of Galaxy 26.1's, with the same inputs and the same expected
 * answer, so a drift between the two implementations shows up as a failure here rather than
 * as two surfaces quietly recommending different containers:
 *
 *  - `v2_image_name`'s eight doctests              (lib/galaxy/tool_util/deps/mulled/util.py)
 *  - `version_sorted` and the two tag selectors    (test/unit/tool_util/mulled/test_mulled_util.py)
 *  - `CondaTarget`'s validation                    (lib/galaxy/tool_util/deps/conda_util.py)
 *  - `recommend_container` and `biocontainer_tag_built`
 *                                                  (test/unit/tool_util/mulled/test_recommend.py)
 *
 * Galaxy's recommender tests fake `mulled_tags_for`, which is a tag list rather than a
 * response, so the fixtures here are written one layer lower -- as the quay.io JSON the
 * port's own fetch and parse have to get through -- with the tag sets taken from those
 * cases. `quayRepo` is shaped after a real
 * `GET https://quay.io/api/v1/repository/biocontainers/<pkg>` body, trimmed to the fields
 * this code reads plus enough of the rest to keep the shape honest.
 *
 * No test here touches the network. Every stub counts the requests it is asked for and the
 * cases that must not reach quay.io assert that count is zero -- a stub that only throws
 * cannot tell a forbidden request from no request at all, because the throw comes back as
 * the same failed lookup the case is asserting.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { GalaxyValidationError } from "../src/errors";
import {
  QUAY_BIOCONTAINERS_PREFIX,
  QUAY_TIMEOUT_MS,
  QuayHttpStatusError,
  QuayMalformedResponseError,
  QuayRequestError,
  __clearRecommendationCacheForTest,
  biocontainerTagBuilt,
  buildTarget,
  encodingFromContentType,
  fetchWithInactivityTimeout,
  quayTagsFor,
  readTagNamesInResponseOrder,
  recommendContainer,
  requestsResponseJson,
  selectMulledV2Tag,
  selectSinglePackageTag,
  splitTag,
  v2ImageName,
  v2RepoName,
  v2VersionHash,
  versionSorted,
  type PackageSpec,
} from "../src/mulled";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/**
 * A quay.io repository body carrying exactly these tags, shaped like the real one -- and built
 * as TEXT rather than as an object handed to `JSON.stringify`.
 *
 * The order of the tag map is part of the fixture. `JSON.stringify` writes an object's
 * integer-shaped keys first, in ascending numeric order, so a fixture round-tripped through it
 * can never present `{"1.0": ..., "1": ...}` in the order quay.io sent them -- which is exactly
 * the case the response-order read exists for, and exactly what the object fixtures used to
 * hide.
 */
function quayRepo(tags: readonly string[]): string {
  const entries = tags.map(
    (tag, i) =>
      `${JSON.stringify(tag)}:{"name":${JSON.stringify(tag)},"size":${1024 * (i + 1)},` +
      `"last_modified":"Mon, 01 Jan 2024 00:00:00 -0000",` +
      `"manifest_digest":"sha256:${String(i).padStart(64, "0")}"}`,
  );
  return (
    '{"namespace":"biocontainers","name":"fixture","kind":"image","description":null,' +
    '"is_public":true,"is_starred":false,"status_token":"","trust_enabled":false,' +
    '"tag_expiration_s":1209600,"can_write":false,"can_admin":false,' +
    `"tags":{${entries.join(",")}}}`
  );
}

/**
 * A real `Response`, not a stand-in: the port streams the body through a reader so it can
 * time out on inactivity rather than on elapsed time, and a hand-rolled `{ json() }` object
 * would leave that path -- the one quay.io actually goes through -- untested.
 */
const okText = (text: string) =>
  new Response(text, { status: 200, headers: { "content-type": "application/json" } });
const okBody = (text: string) => new Response(text, { status: 200 });
const status = (code: number) => new Response("{}", { status: code });
/** A status with a body, for the 3xx that raise_for_status lets through. */
const statusWithBody = (code: number, text: string) => new Response(text, { status: code });

/**
 * A status outside the range `new Response` will build. The constructor refuses anything but
 * 200-599, where the status line on the wire is three digits and llhttp will parse a 600 --
 * so a response undici can hand over is one this file cannot construct. Shadowing the
 * prototype's getter on a real `Response` keeps the streaming body real and changes only the
 * number `quayTagsFor` reads.
 */
function statusOutsideRange(code: number, text: string): Response {
  const resp = new Response(text, { status: 200 });
  Object.defineProperty(resp, "status", { value: code, configurable: true });
  return resp;
}

/** The package name in `/api/v1/repository/biocontainers/<name>`. */
const repoFromUrl = (url: string) => url.slice(url.lastIndexOf("/") + 1);

/**
 * What `fetch` makes of the URL it is handed, which is the request quay.io would see.
 *
 * The string the caller builds is not it: `fetch` parses it under the URL Standard first, and
 * that parse rewrites a path -- a backslash becomes a separator, `.` and `..` segments
 * collapse. A stub asserting on the argument cannot see any of that, so every stub here
 * records the round trip instead.
 */
const asRequested = (url: string) => new URL(url).href;

/**
 * Stub `fetch` with a tag list per repository; anything unlisted answers an empty repo.
 *
 * The table is read through a `Map`, because one of the cases below asks for a repository
 * called `__proto__` and a plain object would answer that with `Object.prototype`.
 */
function stubQuay(byRepo: Record<string, readonly string[]>, asked: string[] = []): string[] {
  const table = new Map(Object.entries(byRepo));
  vi.stubGlobal("fetch", async (url: string) => {
    const requested = asRequested(url);
    asked.push(requested);
    return okText(quayRepo(table.get(repoFromUrl(requested)) ?? []));
  });
  return asked;
}

/** Stub one body for every repository, and count the requests it was asked for. */
function stubBody(make: () => Response): { count: number } {
  const calls = { count: 0 };
  vi.stubGlobal("fetch", async () => {
    calls.count++;
    return make();
  });
  return calls;
}

/**
 * Stub a response whose headers arrive and whose body then tears -- the connection dropping
 * mid-transfer, which is what `requests` reports as a ChunkedEncodingError. The status is the
 * caller's, so a case can put a torn body behind a 404 and see which of the two Python
 * notices first.
 */
function stubTornBody(code: number): { count: number } {
  const calls = { count: 0 };
  vi.stubGlobal("fetch", async () => {
    calls.count++;
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode('{"tags":{"1.17'));
        controller.error(new Error("socket hang up"));
      },
    });
    return new Response(body, { status: code });
  });
  return calls;
}

/**
 * Stub a `fetch` that must never be called. It counts first and throws second, so a case
 * that forbids a request fails on the count even though the throw would have been folded
 * into exactly the "not found" / "lookup failed" answer the case asserts.
 */
function stubNoNetwork(): { count: number } {
  const calls = { count: 0 };
  vi.stubGlobal("fetch", async () => {
    calls.count++;
    throw new Error("the network must not be consulted here");
  });
  return calls;
}

const single = (name: string, version?: string): PackageSpec[] => [{ name, version }];
const target = (name: string, version?: string, build?: string) => buildTarget(name, version, build);
const API = "https://quay.io/api/v1/repository";
const repoUrl = (repo: string) => `${API}/biocontainers/${repo}`;

beforeEach(() => {
  __clearRecommendationCacheForTest();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  vi.useRealTimers();
  __clearRecommendationCacheForTest();
});

// ---------------------------------------------------------------------------
// v2_image_name -- Galaxy's eight doctests, reproduced hash for hash
// ---------------------------------------------------------------------------

describe("v2ImageName", () => {
  // Six distinct strings cover all eight doctests: the two that pass image_build="0" expect
  // the same strings as the two above them, because _simple_image_name special-cases "0"
  // back to no suffix at all, and the recommender never passes an image_build anyway.
  it("names a single target by package and version, unhashed", () => {
    expect(v2ImageName([target("samtools", "1.3.1")])).toBe("samtools:1.3.1");
  });

  it("appends a conda build string to a single target", () => {
    expect(v2ImageName([target("samtools", "1.3.1", "py_1")])).toBe("samtools:1.3.1--py_1");
  });

  it("hashes names and versions for several targets", () => {
    expect(v2ImageName([target("samtools", "1.3.1"), target("bwa", "0.7.13")])).toBe(
      "mulled-v2-fe8faa35dbf6dc65a0f7f5d4ea12e31a79f73e40:4d0535c94ef45be8459f429561f0894c3fe0ebcf",
    );
  });

  it("counts an unpinned package as the literal 'null' in the version hash", () => {
    expect(v2ImageName([target("samtools", "1.3.1"), target("bwa")])).toBe(
      "mulled-v2-fe8faa35dbf6dc65a0f7f5d4ea12e31a79f73e40:b0c847e4fb89c343b04036e33b2daa19c4152cf5",
    );
  });

  it("drops the version hash entirely when nothing is pinned", () => {
    expect(v2ImageName([target("samtools"), target("bwa")])).toBe(
      "mulled-v2-fe8faa35dbf6dc65a0f7f5d4ea12e31a79f73e40",
    );
  });

  it("keeps build strings out of both hashes", () => {
    expect(
      v2ImageName([target("samtools", "1.3.1", "h9071d68_10"), target("bedtools", "2.26.0", "0")]),
    ).toBe(
      "mulled-v2-8186960447c5cb2faa697666dc1e6d919ad23f3e:a6419f25efff953fc505dbd5ee734856180bb619",
    );
  });

  it("hashes by package name order, not argument order", () => {
    const a = v2ImageName([target("samtools", "1.3.1"), target("bwa", "0.7.13")]);
    const b = v2ImageName([target("bwa", "0.7.13"), target("samtools", "1.3.1")]);
    expect(a).toBe(b);
  });

  it("orders the names by code point, so the hash is Python's", () => {
    // Python's sorted() puts U+E000 before U+10000; JavaScript's `<` puts it after, because
    // an astral character is stored as a surrogate pair starting at U+D800. The order is
    // what goes into both hashes, so getting it wrong asks quay.io about another image.
    // Both strings below are the image v2_image_name produced in the Galaxy checkout.
    const astral = "\u{10000}";
    const pua = "";
    expect(v2ImageName([target(astral, "1"), target(pua, "2")])).toBe(
      "mulled-v2-32b2ca5b807e3bc17994b51d4f72768710de5cef:30794c5408df18027637b235fe907e00ac7cec4f",
    );
    expect(v2ImageName([target(pua, "2"), target(astral, "1")])).toBe(
      "mulled-v2-32b2ca5b807e3bc17994b51d4f72768710de5cef:30794c5408df18027637b235fe907e00ac7cec4f",
    );
    expect(v2RepoName([astral, pua])).toBe(
      "mulled-v2-32b2ca5b807e3bc17994b51d4f72768710de5cef",
    );
  });

  it("splits into the repo name and the version hash the way the recommender reads it", () => {
    const specs = [{ name: "samtools", version: "1.3.1" }, { name: "bwa", version: "0.7.13" }];
    expect(v2RepoName(specs.map((t) => t.name))).toBe(
      "mulled-v2-fe8faa35dbf6dc65a0f7f5d4ea12e31a79f73e40",
    );
    expect(v2VersionHash(specs)).toBe("4d0535c94ef45be8459f429561f0894c3fe0ebcf");
    expect(v2VersionHash([{ name: "samtools" }, { name: "bwa" }])).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// build_target / CondaTarget -- conda_util.py
// ---------------------------------------------------------------------------

describe("buildTarget", () => {
  it("lowercases the package name, as CondaTarget does", () => {
    expect(target("SamTools", "1.3.1")).toEqual({
      package: "samtools",
      version: "1.3.1",
      build: null,
    });
  });

  it.each([
    ["sam tools", "Invalid package [sam tools] encountered."],
    ['sam"tools', 'Invalid package [sam"tools] encountered.'],
    ["sam'tools", "Invalid package [sam'tools] encountered."],
    ["", "Invalid package [] encountered."],
  ])("refuses the shell-unsafe package name %j", (name, message) => {
    expect(() => target(name, "1.0")).toThrow(message);
  });

  it.each([
    ["1.17 2", "Invalid version [1.17 2] encountered."],
    ["1.17'", "Invalid version [1.17'] encountered."],
    ['1.17"', 'Invalid version [1.17"] encountered.'],
    ["1.17\t2", "Invalid version [1.17\t2] encountered."],
  ])("refuses the shell-unsafe version %j", (version, message) => {
    expect(() => target("samtools", version)).toThrow(message);
  });

  it("refuses a shell-unsafe build string", () => {
    expect(() => target("samtools", "1.17", "py_1 x")).toThrow(
      "Invalid build [py_1 x] encountered.",
    );
  });

  it.each(["\u0085", "\u001c", "\u001d", "\u001e", "\u001f", "\u00a0", "\u2028", "\u3000"])(
    "refuses a version carrying %j, which Python's \\s matches and JavaScript's may not",
    (space) => {
      expect(() => target("samtools", `1.17${space}2`)).toThrow(
        `Invalid version [1.17${space}2] encountered.`,
      );
    },
  );

  it("allows a U+FEFF, which Python's \\s does not match", () => {
    // It survives pyStrip too, so Python carries it into the version it looks for -- and then
    // finds no tag for it. Refusing it here would have been a different answer entirely.
    expect(target("samtools", "\ufeff1.17")).toEqual({
      package: "samtools",
      version: "\ufeff1.17",
      build: null,
    });
  });

  it("allows an empty version and an empty build, which are not the same as unsafe", () => {
    // Python's `if version and ...` skips the check for "", and `if build is not None`
    // runs it but "" carries nothing unsafe.
    expect(target("samtools", "", "")).toEqual({ package: "samtools", version: "", build: "" });
  });
});

// ---------------------------------------------------------------------------
// split_tag / version_sorted -- test_mulled_util.py
// ---------------------------------------------------------------------------

describe("splitTag", () => {
  it("takes everything before the LAST separator, as rsplit does", () => {
    expect(splitTag("1.17--h00cdaf9_0")).toBe("1.17");
    expect(splitTag("1.17")).toBe("1.17");
    // A version that itself contains the separator stays whole, which a plain split loses.
    expect(splitTag("1--2--h0_0")).toBe("1--2");
  });
});

describe("versionSorted", () => {
  it.each([
    [["2.22--he941832_1", "2.22--he860b03_2", "2.22--hdbcaa40_3"], "2.22--hdbcaa40_3"],
    [["1.1.2--py27_0", "1.1.2--py36_0", "1.1.2--py35_0"], "1.1.2--py36_0"],
    [
      ["6725cda82000b8e514baddcbf8c2dce054e3f797-1", "6725cda82000b8e514baddcbf8c2dce054e3f797-0"],
      "6725cda82000b8e514baddcbf8c2dce054e3f797-1",
    ],
    [["python:3.5", "python:3.7", "python:3.7--2"], "python:3.7--2"],
  ])("puts the newest of %j first", (tags, newest) => {
    expect(versionSorted(tags)[0]).toBe(newest);
  });

  it("orders numerically rather than lexically", () => {
    expect(versionSorted(["1.9--h0_0", "1.10--h0_0", "1.17--h0_0"])[0]).toBe("1.17--h0_0");
  });

  it("does not stop at a shared prefix, so 1.17.1 beats 1.17", () => {
    // The case a hand-rolled comparator got wrong before this went through parse_version:
    // both tags have the same build number, so the release segment decides.
    expect(versionSorted(["1.17--h0_0", "1.17.1--h0_0"])).toEqual(["1.17.1--h0_0", "1.17--h0_0"]);
    expect(versionSorted(["1.17.1--h0_0", "1.17--h0_0"])).toEqual(["1.17.1--h0_0", "1.17--h0_0"]);
  });

  it("reads a post-release as newer than the release it follows", () => {
    expect(versionSorted(["1.0--h0_0", "1.0.post1--h0_0"])[0]).toBe("1.0.post1--h0_0");
  });

  it("treats 1.0 and 1.0.0 as the same version, so the build number decides", () => {
    expect(versionSorted(["1.0--h0_1", "1.0.0--h0_2"])[0]).toBe("1.0.0--h0_2");
    expect(versionSorted(["1.0--h0_3", "1.0.0--h0_2"])[0]).toBe("1.0--h0_3");
  });

  it("orders release components past 2^53 exactly, which a JS number cannot", () => {
    // 9007199254740992 and ...993 are the same double, so a number-based comparator calls
    // these two tags equal and version_sorted leaves them in whatever order quay.io listed.
    const pair = ["9007199254740993--h0_0", "9007199254740992--h0_0"];
    expect(versionSorted(pair)).toEqual(pair);
    expect(versionSorted([...pair].reverse())).toEqual(pair);
  });

  it("orders a 30-digit release component", () => {
    const big = `1.${"9".repeat(30)}--h0_0`;
    const smaller = `1.${"9".repeat(29)}8--h0_0`;
    expect(versionSorted([smaller, big])).toEqual([big, smaller]);
    expect(versionSorted([big, smaller])).toEqual([big, smaller]);
  });

  it("orders build numbers past 2^53 exactly too", () => {
    // The build-number pass runs after the build-string pass and overrides it, so these two
    // have to disagree for the number to be what decides: "a_..." sorts under "b_..." as a
    // build string, and above it as a build number. Read as doubles the two numbers are one
    // value, the pass is a tie, and the build-string order survives instead.
    const pair = ["1.0--a_9007199254740993", "1.0--b_9007199254740992"];
    expect(versionSorted(pair)).toEqual(pair);
    expect(versionSorted([...pair].reverse())).toEqual(pair);
  });

  it("puts a tag that carries a build above the same version without one", () => {
    // parse_tag forces build_number back to -1 for a bare version, and its "-1" build-string
    // placeholder is a legacy version, which sorts under every real build string.
    expect(versionSorted(["1.17", "1.17--h0_0"])[0]).toBe("1.17--h0_0");
  });

  it("reads a tag's whitespace and case the way Python's regexes do", () => {
    // Each of these is version_sorted's own answer in the Galaxy checkout. A tag name is
    // whatever quay.io put in the map, so it reaches parse_version unstripped and the
    // pattern's `^\s*` and IGNORECASE are what decide whether it is a version at all.
    expect(versionSorted(["1.0", "0.9"])).toEqual(["1.0", "0.9"]);
    expect(versionSorted(["﻿1.0", "0.9"])).toEqual(["0.9", "﻿1.0"]);
    expect(versionSorted(["1.0+K", "0.9"])).toEqual(["1.0+K", "0.9"]);
    // Equal versions, so the tie leaves them in the response order.
    expect(versionSorted(["1.0+K", "1.0+k"])).toEqual(["1.0+K", "1.0+k"]);
  });

  it("orders tag names by code point, as a Python string comparison does", () => {
    // Through the legacy key, whose parts are slices of the tag name: U+E000 sorts below
    // U+10000 for Python and above it for JavaScript.
    expect(versionSorted(["x", "x\u{10000}"])).toEqual(["x\u{10000}", "x"]);
    expect(versionSorted(["x\u{10000}", "x"])).toEqual(["x\u{10000}", "x"]);
  });
});

// ---------------------------------------------------------------------------
// The tag selectors -- test_mulled_util.py
// ---------------------------------------------------------------------------

describe("selectSinglePackageTag", () => {
  it("matches the requested version exactly", () => {
    expect(selectSinglePackageTag(["1.17--h0_0", "1.16--h0_0"], "1.16")).toEqual({
      tag: "1.16--h0_0",
      exact: true,
    });
  });

  it("returns nothing when the version is absent and no fallback is allowed", () => {
    expect(selectSinglePackageTag(["1.17--h0_0"], "9.9")).toEqual({ tag: null, exact: false });
  });

  it("falls back to the newest tag when asked to", () => {
    expect(selectSinglePackageTag(["1.17--h0_0"], "9.9", { allowNewestFallback: true })).toEqual({
      tag: "1.17--h0_0",
      exact: false,
    });
  });

  it("needs the fallback even with no version requested", () => {
    expect(selectSinglePackageTag(["1.17--h0_0"], null)).toEqual({ tag: null, exact: false });
    expect(selectSinglePackageTag(["1.17--h0_0"], null, { allowNewestFallback: true })).toEqual({
      tag: "1.17--h0_0",
      exact: false,
    });
  });

  it("has nothing to pick from an empty tag list", () => {
    expect(selectSinglePackageTag([], "1.0")).toEqual({ tag: null, exact: false });
  });
});

describe("selectMulledV2Tag", () => {
  it("picks the newest build carrying the requested version hash", () => {
    expect(selectMulledV2Tag(["abc123-1", "abc123-0", "def456-0"], "abc123")).toEqual({
      tag: "abc123-1",
      exact: true,
    });
  });

  it("returns nothing for an unbuilt hash when no fallback is allowed", () => {
    expect(selectMulledV2Tag(["def456-0"], "abc123")).toEqual({ tag: null, exact: false });
  });

  it("falls back to the newest tag when asked to", () => {
    expect(selectMulledV2Tag(["def456-0"], "abc123", { allowNewestFallback: true })).toEqual({
      tag: "def456-0",
      exact: false,
    });
  });

  it("counts the newest tag as exact when there is no version hash to mismatch", () => {
    expect(selectMulledV2Tag(["def456-0"], null)).toEqual({ tag: "def456-0", exact: true });
  });

  it("has nothing to pick from an empty tag list", () => {
    expect(selectMulledV2Tag([], "abc123")).toEqual({ tag: null, exact: false });
  });
});

// ---------------------------------------------------------------------------
// The socket timeout -- requests' timeout= is inactivity, not elapsed time
// ---------------------------------------------------------------------------

/** A 200 whose headers arrive after `headerMs` and whose body then comes in `chunks`, `gapMs` apart. */
function stubChunkedBody(
  text: string,
  chunks: number,
  gapMs: number,
  headerMs = 0,
): { count: number } {
  const calls = { count: 0 };
  const size = Math.ceil(text.length / chunks);
  vi.stubGlobal("fetch", async () => {
    calls.count++;
    if (headerMs > 0) await new Promise((resolve) => setTimeout(resolve, headerMs));
    let sent = 0;
    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        return new Promise<void>((resolve) => {
          setTimeout(() => {
            if (sent >= text.length) controller.close();
            else {
              controller.enqueue(new TextEncoder().encode(text.slice(sent, sent + size)));
              sent += size;
            }
            resolve();
          }, gapMs);
        });
      },
    });
    return new Response(body, { status: 200 });
  });
  return calls;
}

describe("fetchWithInactivityTimeout", () => {
  it("is the 12s Galaxy uses, and quayTagsFor is its only caller", () => {
    expect(QUAY_TIMEOUT_MS).toBe(12_000);
  });

  it("lets a body that takes longer than the timeout in total through", async () => {
    // Eight chunks 10ms apart is 80ms of streaming against a 50ms timeout. requests would
    // read this happily; an AbortSignal.timeout deadline would abort it at 50ms.
    const payload = quayRepo(["1.17--h0_0", "1.16--h0_0"]);
    stubChunkedBody(payload, 8, 10);
    const out = await fetchWithInactivityTimeout(repoUrl("samtools"), 50);
    expect(out.status).toBe(200);
    // Bytes now, because the charset is the caller's decision -- see requestsResponseJson.
    expect(new TextDecoder().decode(out.body)).toBe(payload);
  });

  it("waits two timeouts for the headers, as requests budgets connect and read apart", async () => {
    // requests spends up to `timeout` connecting and then a fresh `timeout` on the first read,
    // so a 7s connect followed by 7s of silence is a healthy response at timeout=12. In ms:
    // headers at 45ms against a 30ms timeout, which a one-timeout budget would have aborted.
    const started = Date.now();
    stubChunkedBody(quayRepo(["1.17--h0_0"]), 1, 0, 45);
    const out = await fetchWithInactivityTimeout(repoUrl("samtools"), 30);
    expect(out.status).toBe(200);
    expect(Date.now() - started).toBeGreaterThanOrEqual(40);
  });

  it("gives up once the headers are past two timeouts", async () => {
    stubChunkedBody(quayRepo(["1.17--h0_0"]), 1, 0, 200);
    const err = await fetchWithInactivityTimeout(repoUrl("samtools"), 30).catch(
      (e: unknown) => e as Error,
    );
    expect(err).toBeInstanceOf(QuayRequestError);
    expect(err.message).toContain("Read timed out.");
  });

  it("still holds the body to one timeout between chunks after a slow start", async () => {
    // Headers at 60ms get through the 80ms header budget; the 120ms gap that follows does not
    // get through the 40ms read budget. Both halves, in one case.
    stubChunkedBody(quayRepo(["1.17--h0_0"]), 4, 120, 60);
    const err = await fetchWithInactivityTimeout(repoUrl("samtools"), 40).catch(
      (e: unknown) => e as Error,
    );
    expect(err).toBeInstanceOf(QuayRequestError);
    expect(err.message).toContain("Read timed out.");
  });

  it("gives up when the stream goes quiet for longer than the timeout", async () => {
    stubChunkedBody(quayRepo(["1.17--h0_0"]), 4, 120);
    const err = await fetchWithInactivityTimeout(repoUrl("samtools"), 40).catch(
      (e: unknown) => e as Error,
    );
    expect(err).toBeInstanceOf(QuayRequestError);
    expect(err.message).toContain("Read timed out.");
  });

  it("gives up when the response never starts", async () => {
    vi.stubGlobal("fetch", async (_url: string, init: { signal: AbortSignal }) => {
      await new Promise((_resolve, reject) => {
        init.signal.addEventListener("abort", () => reject(init.signal.reason as Error));
      });
      throw new Error("unreachable");
    });
    // Two budgets now, so 15ms rather than 30ms keeps the case at the same wall-clock cost.
    await expect(fetchWithInactivityTimeout(repoUrl("samtools"), 15)).rejects.toBeInstanceOf(
      QuayRequestError,
    );
  });

  it("turns a slow stream into a failed lookup rather than a wrong answer", async () => {
    stubChunkedBody(quayRepo(["1.17--h0_0"]), 4, 120);
    // Through quayTagsFor the timeout is 12s, which no test waits for -- but the failure
    // mode is what matters here: a QuayRequestError, so recommendContainer degrades.
    const err = await fetchWithInactivityTimeout(repoUrl("samtools"), 40)
      .then(() => null)
      .catch((e: unknown) => e);
    expect(err).toBeInstanceOf(QuayRequestError);
    expect(err).not.toBeInstanceOf(QuayHttpStatusError);
  });
});

// ---------------------------------------------------------------------------
// The tag names, in the order quay.io sent them
// ---------------------------------------------------------------------------

describe("readTagNamesInResponseOrder", () => {
  it("keeps integer-shaped keys where the response put them, which Object.keys does not", () => {
    const text = '{"tags":{"1.0":{},"1":{}}}';
    expect(readTagNamesInResponseOrder(text)).toEqual(["1.0", "1"]);
    // What the port used to read, and what a stringified fixture would still hide: JavaScript
    // enumerates "1" before "1.0" because it is an array index, and re-serialising says so.
    const parsed = JSON.parse(text) as { tags: Record<string, unknown> };
    expect(Object.keys(parsed.tags)).toEqual(["1", "1.0"]);
    expect(JSON.stringify(parsed)).toBe('{"tags":{"1":{},"1.0":{}}}');
  });

  it("reads the other order as the other order", () => {
    expect(readTagNamesInResponseOrder('{"tags":{"1":{},"1.0":{}}}')).toEqual(["1", "1.0"]);
  });

  it("steps over values without interpreting them", () => {
    const text =
      '{"description":"a } brace and a \\" quote","tags":{"1.17--h0_0":{"name":"x{y","size":7}},' +
      '"is_public":true,"tag_expiration_s":1209600}';
    expect(readTagNamesInResponseOrder(text)).toEqual(["1.17--h0_0"]);
  });

  it("takes the LAST tags member, as json.loads and JSON.parse both do", () => {
    const text = '{"tags":{"a-0":{}},"tags":{"b-1":{}}}';
    expect(readTagNamesInResponseOrder(text)).toEqual(["b-1"]);
    expect(Object.keys((JSON.parse(text) as { tags: object }).tags)).toEqual(["b-1"]);
  });

  it("collapses a repeated tag name onto its first position, as a dict does", () => {
    expect(readTagNamesInResponseOrder('{"tags":{"1.0":{},"2.0":{},"1.0":{}}}')).toEqual([
      "1.0",
      "2.0",
    ]);
  });

  it("decodes an escaped key", () => {
    expect(readTagNamesInResponseOrder('{"tags":{"1.0\\u002d0":{}}}')).toEqual(["1.0-0"]);
  });

  it("does not mind whitespace between the tokens", () => {
    expect(readTagNamesInResponseOrder('{ "tags" : { "1.0" : { } , "1" : { } } }')).toEqual([
      "1.0",
      "1",
    ]);
  });

  it("reads an empty tag map as no tags, not as a missing one", () => {
    expect(readTagNamesInResponseOrder('{"tags":{}}')).toEqual([]);
  });

  it.each(['{"tags":[]}', '{"tags":null}', '{"tags":7}', '{"tags":"x"}', "[]", "42", "{}"])(
    "answers null for %s, which is not a tag mapping",
    (text) => {
      expect(readTagNamesInResponseOrder(text)).toBeNull();
    },
  );
});

// ---------------------------------------------------------------------------
// quayTagsFor -- the one endpoint this module calls
// ---------------------------------------------------------------------------

describe("quayTagsFor", () => {
  it("asks the biocontainers repository endpoint and drops 'latest'", async () => {
    const asked: string[] = [];
    stubQuay({ samtools: ["1.17--h0_0", "latest", "1.16--h0_0"] }, asked);
    expect(await quayTagsFor("samtools")).toEqual(["1.17--h0_0", "1.16--h0_0"]);
    expect(asked).toEqual([repoUrl("samtools")]);
  });

  it("returns the tags newest-first whatever order quay.io listed them in", async () => {
    stubQuay({ samtools: ["1.16--h0_0", "1.17--h0_0", "1.9--h0_0"] });
    expect(await quayTagsFor("samtools")).toEqual(["1.17--h0_0", "1.16--h0_0", "1.9--h0_0"]);
  });

  it("keeps the response order for tags version_sorted cannot tell apart", async () => {
    // "1" and "1.0" are one version to parse_version and both end up with build number -1,
    // so the sort is a tie and the response order decides which tag is "newest".
    stubBody(() => okText('{"tags":{"1.0":{},"1":{}}}'));
    expect(await quayTagsFor("samtools")).toEqual(["1.0", "1"]);
  });

  it("parses the body of a 3xx, which raise_for_status does not refuse", async () => {
    // requests rejects only 4xx and 5xx, so a 302 with no Location -- which fetch does not
    // follow either -- is a body Python goes on to read rather than a missing repository.
    const calls = stubBody(() => statusWithBody(302, '{"tags":{"1.17--h0_0":{}}}'));
    expect(await quayTagsFor("samtools")).toEqual(["1.17--h0_0"]);
    expect(calls.count).toBe(1);
  });

  it("reads an invalid_token body as an empty repository", async () => {
    const calls = stubBody(() => okText('{"error_type":"invalid_token","error_message":"no"}'));
    expect(await quayTagsFor("samtools")).toEqual([]);
    expect(calls.count).toBe(1);
  });

  it("turns a 4xx or a 5xx into the HTTP-status error, not just a 404", async () => {
    for (const code of [400, 404, 500, 599]) {
      const calls = stubBody(() => status(code));
      await expect(quayTagsFor("nopackage")).rejects.toBeInstanceOf(QuayHttpStatusError);
      expect(calls.count).toBe(1);
    }
  });

  it("parses the body of a status outside 400-599, which is all raise_for_status tests", async () => {
    // Two range tests, not "not ok" and not ">= 400". 399 and 600 are both status lines a
    // server can send, and Python reads the body of each.
    const body = '{"tags":{"1.17--h0_0":{}}}';
    const below = stubBody(() => statusWithBody(399, body));
    expect(await quayTagsFor("samtools")).toEqual(["1.17--h0_0"]);
    expect(below.count).toBe(1);

    const above = stubBody(() => statusOutsideRange(600, body));
    expect(await quayTagsFor("samtools")).toEqual(["1.17--h0_0"]);
    expect(above.count).toBe(1);
  });

  it("calls a 4xx whose body will not finish a failed request, as requests does", async () => {
    // `requests` reads the body inside send() and only the caller's raise_for_status()
    // afterwards looks at the status, so a truncated body on a 404 raises
    // ChunkedEncodingError -- a RequestException -- and never reaches the HTTPError branch.
    // The difference is visible from outside: "lookup failed: ..." rather than "no
    // biocontainer found", which is a transient failure rather than a missing repository.
    const calls = stubTornBody(404);
    const err = await quayTagsFor("samtools").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(QuayRequestError);
    expect(err).not.toBeInstanceOf(QuayHttpStatusError);
    expect(calls.count).toBe(1);

    stubTornBody(404);
    const out = await recommendContainer(single("samtools"));
    expect(out.notes[0]).toMatch(/^lookup failed: /);
    expect(out.found).toBe(false);
  });

  it("turns a transport failure into the request error", async () => {
    const calls = stubNoNetwork();
    const err = await quayTagsFor("samtools").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(QuayRequestError);
    expect(err).not.toBeInstanceOf(QuayHttpStatusError);
    expect(calls.count).toBe(1);
  });

  it("turns a body that is not JSON into the request error, as requests does", async () => {
    stubBody(() => okBody("<html>504 gateway timeout</html>"));
    await expect(quayTagsFor("samtools")).rejects.toBeInstanceOf(QuayRequestError);
  });
});

// ---------------------------------------------------------------------------
// quayTagsFor -- a 200 whose body is not the shape Python reads
// ---------------------------------------------------------------------------

describe("quayTagsFor (a malformed 200)", () => {
  it.each([
    ['{"namespace":"biocontainers"}', "no tags description found"],
    ['["1.17--h0_0"]', "no tags description found"],
    ['"biocontainers/samtools"', "no tags description found"],
    ["42", "no tags description found"],
    ["null", "no tags description found"],
    ['{"tags":["1.17--h0_0"]}', "the tag list is not a mapping"],
    ['{"tags":null}', "the tag list is not a mapping"],
    ['{"tags":"1.17--h0_0"}', "the tag list is not a mapping"],
    ['{"tags":7}', "the tag list is not a mapping"],
  ])("refuses %s", async (body, message) => {
    stubBody(() => okText(body));
    const err = await quayTagsFor("samtools").catch((e: unknown) => e as Error);
    expect(err).toBeInstanceOf(QuayMalformedResponseError);
    // Not a QuayRequestError, so recommendContainer lets it out instead of degrading.
    expect(err).not.toBeInstanceOf(QuayRequestError);
    expect(err.message).toContain(message);
  });

  it("never fabricates a tag out of a list-shaped tag map", async () => {
    // Object.keys(["1.17--h0_0"]) is ["0"], which a lenient read turns into the tag "0" --
    // and then into a found, verified `samtools:0` that was never built. Python's
    // `.keys()` raises on a list, and so does this.
    stubBody(() => okText('{"tags":["1.17--h0_0"]}'));
    await expect(recommendContainer(single("samtools"))).rejects.toBeInstanceOf(
      QuayMalformedResponseError,
    );
    await expect(
      biocontainerTagBuilt(`${QUAY_BIOCONTAINERS_PREFIX}/samtools:0`),
    ).rejects.toBeInstanceOf(QuayMalformedResponseError);
  });
});

// ---------------------------------------------------------------------------
// recommendContainer, single package -- test_recommend.py
// ---------------------------------------------------------------------------

describe("recommendContainer (single package)", () => {
  it("resolves an exact version", async () => {
    const asked = stubQuay({ samtools: ["1.17--h00cdaf9_0", "1.16--h0_0"] });
    const rec = await recommendContainer(single("samtools", "1.17"));
    expect(rec.image).toBe("quay.io/biocontainers/samtools:1.17--h00cdaf9_0");
    expect(rec.source).toBe("quay_single");
    expect(rec.match_quality).toBe("exact_version");
    expect(rec.multi_package).toBe(false);
    expect(rec.found).toBe(true);
    expect(asked).toEqual([repoUrl("samtools")]);
  });

  it("falls back to the newest tag, with a note, when the version is missing", async () => {
    stubQuay({ samtools: ["1.17--h0_0"] });
    const rec = await recommendContainer(single("samtools", "9.9"));
    expect(rec.image).toBe("quay.io/biocontainers/samtools:1.17--h0_0");
    expect(rec.match_quality).toBe("name_only");
    expect(rec.notes).toEqual([
      "requested version 9.9 not found; newest available tag is 1.17--h0_0",
    ]);
  });

  it("is name_only, and silent, when no version was asked for", async () => {
    stubQuay({ samtools: ["1.17--h0_0", "1.16--h0_0"] });
    const rec = await recommendContainer(single("samtools"));
    expect(rec.image).toBe("quay.io/biocontainers/samtools:1.17--h0_0");
    expect(rec.match_quality).toBe("name_only");
    expect(rec.notes).toEqual([]);
  });

  it("picks the newest tag by PEP 440, not by text", async () => {
    // Two tags with the same build number: 1.17.1 is the newest, and reading the tags as
    // text would answer 1.17 instead.
    stubQuay({ samtools: ["1.17--h0_0", "1.17.1--h0_0"] });
    const rec = await recommendContainer(single("samtools"));
    expect(rec.image).toBe("quay.io/biocontainers/samtools:1.17.1--h0_0");
  });

  it("lowercases and trims the package name before asking quay.io", async () => {
    const asked = stubQuay({ samtools: ["1.17--h0_0"] });
    const rec = await recommendContainer([{ name: "  SamTools  ", version: " 1.17 " }]);
    expect(asked).toEqual([repoUrl("samtools")]);
    expect(rec.image).toBe("quay.io/biocontainers/samtools:1.17--h0_0");
    expect(rec.match_quality).toBe("exact_version");
  });

  it("reports a missing repository as not found rather than as a failure", async () => {
    const calls = stubBody(() => status(404));
    const rec = await recommendContainer(single("nopackage", "1.0"));
    expect(rec.image).toBeNull();
    expect(rec.source).toBe("none");
    expect(rec.match_quality).toBe("not_found");
    expect(rec.notes).toEqual(["no biocontainer found for 'nopackage'"]);
    expect(rec.found).toBe(false);
    expect(calls.count).toBe(1);
  });

  it("reports a network error as a lookup failure", async () => {
    const calls = stubNoNetwork();
    const rec = await recommendContainer(single("samtools", "1.0"));
    expect(rec.image).toBeNull();
    expect(rec.source).toBe("none");
    expect(rec.notes.some((n) => n.includes("lookup failed"))).toBe(true);
    expect(calls.count).toBe(1);
  });

  it("refuses a shell-unsafe version before asking quay.io anything", async () => {
    // Python's build_target constructs a CondaTarget, which validates first, so this is a
    // refusal rather than a quietly successful newest-tag fallback.
    const calls = stubNoNetwork();
    await expect(recommendContainer(single("samtools", "1.17 2"))).rejects.toThrow(
      "Invalid version [1.17 2] encountered.",
    );
    expect(calls.count).toBe(0);
  });

  it("refuses a shell-unsafe package name before asking quay.io anything", async () => {
    const calls = stubNoNetwork();
    await expect(recommendContainer(single("sam'tools", "1.17"))).rejects.toThrow(
      "Invalid package [sam'tools] encountered.",
    );
    expect(calls.count).toBe(0);
  });

  it("does not cache a refusal, because the refusal escapes before the cache is written", async () => {
    const calls = stubNoNetwork();
    await expect(recommendContainer(single("samtools", "1.17 2"))).rejects.toThrow();
    await expect(recommendContainer(single("samtools", "1.17 2"))).rejects.toThrow();
    expect(calls.count).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// recommendContainer, several packages -- test_recommend.py
// ---------------------------------------------------------------------------

describe("recommendContainer (several packages)", () => {
  const pinned: PackageSpec[] = [
    { name: "bwa", version: "0.7.17" },
    { name: "samtools", version: "1.17" },
  ];
  const repo = v2RepoName(["bwa", "samtools"]);

  it("resolves an exact version combination to its newest build", async () => {
    const vh = v2VersionHash(pinned)!;
    const asked = stubQuay({ [repo]: [`${vh}-1`, `${vh}-0`, "other-0"] });
    const rec = await recommendContainer(pinned);
    expect(rec.image).toBe(`${QUAY_BIOCONTAINERS_PREFIX}/${repo}:${vh}-1`);
    expect(rec.source).toBe("quay_mulled_v2");
    expect(rec.match_quality).toBe("exact_version");
    expect(rec.multi_package).toBe(true);
    expect(rec.notes).toEqual([]);
    expect(asked).toEqual([repoUrl(repo)]);
  });

  it("falls back to the newest build, with a note, when that combination is not built", async () => {
    stubQuay({ [repo]: ["unrelatedhash-0"] });
    const rec = await recommendContainer(pinned);
    expect(rec.image).toBe(`${QUAY_BIOCONTAINERS_PREFIX}/${repo}:unrelatedhash-0`);
    expect(rec.match_quality).toBe("name_only");
    expect(rec.notes).toEqual([
      "exact version combination not built; using newest available mulled-v2 build",
    ]);
  });

  it("reports an empty mulled-v2 repository as not found", async () => {
    const asked = stubQuay({});
    const rec = await recommendContainer(pinned);
    expect(rec.image).toBeNull();
    expect(rec.source).toBe("none");
    expect(rec.notes).toEqual(["no multi-package biocontainer built for ['bwa', 'samtools']"]);
    expect(asked).toEqual([repoUrl(repo)]);
  });

  it("reports a network error as a lookup failure", async () => {
    const calls = stubNoNetwork();
    const rec = await recommendContainer(pinned);
    expect(rec.image).toBeNull();
    expect(rec.notes.some((n) => n.includes("lookup failed"))).toBe(true);
    expect(calls.count).toBe(1);
  });

  it("refuses a shell-unsafe version in a fully pinned set before asking anything", async () => {
    const calls = stubNoNetwork();
    await expect(
      recommendContainer([{ name: "bwa", version: "0.7.17" }, { name: "samtools", version: "1 2" }]),
    ).rejects.toThrow("Invalid version [1 2] encountered.");
    expect(calls.count).toBe(0);
  });

  it("says so, and looks nothing up, when version resolution is switched off", async () => {
    const asked = stubQuay({ [repo]: ["somehash-0"] });
    const rec = await recommendContainer(
      [{ name: "bwa", version: "0.7.17" }, { name: "samtools" }],
      { resolveVersions: false },
    );
    expect(rec.match_quality).toBe("name_only");
    expect(rec.image).toBe(`${QUAY_BIOCONTAINERS_PREFIX}/${repo}:somehash-0`);
    expect(rec.notes).toEqual(["not all packages were versioned; resolved by package names only"]);
    // Only the mulled-v2 repo, never the unpinned package's candidate versions.
    expect(asked).toEqual([repoUrl(repo)]);
  });

  it("resolves an unpinned package's version against the combinations that are built", async () => {
    const targetHash = v2VersionHash(pinned)!;
    const asked = stubQuay({
      [repo]: [`${targetHash}-0`, "olderhash-0"],
      bwa: ["0.7.18--h0_0", "0.7.17--h0_0", "0.7.16--h0_0"],
    });
    const rec = await recommendContainer([{ name: "samtools", version: "1.17" }, { name: "bwa" }]);
    expect(rec.match_quality).toBe("exact_version");
    expect(rec.image).toBe(`${QUAY_BIOCONTAINERS_PREFIX}/${repo}:${targetHash}-0`);
    expect(rec.notes).toEqual(["resolved version(s): bwa=0.7.17"]);
    // The mulled-v2 repo, then bwa's own tags for its candidate versions. Nothing else.
    expect(asked).toEqual([repoUrl(repo), repoUrl("bwa")]);
  });

  it("resolves a package called __proto__ without losing its version", async () => {
    // A conda package may be called anything, and `resolved["__proto__"] = "1.0"` on a plain
    // object runs Object.prototype's setter instead of adding a property -- so the note came
    // out as `__proto__=[object Object]` where Python's dict just holds the string. Both
    // hashes below are v2_image_name's own, from the Galaxy checkout.
    const protoRepo = "mulled-v2-ed6152c95d76070bf3ac6526037350c28fd3dc1d";
    const targetHash = "a07ebab5a1b76fab57aa52919c619dccc0f4798d";
    const asked = stubQuay({
      [protoRepo]: [`${targetHash}-0`],
      // Computed, because a bare `__proto__:` key in an object literal sets the prototype
      // rather than adding a property -- the same trap one layer up.
      ["__proto__"]: ["1.0--h0_0"],
    });
    const rec = await recommendContainer([{ name: "__proto__" }, { name: "samtools", version: "1.17" }]);
    expect(rec.notes).toEqual(["resolved version(s): __proto__=1.0"]);
    expect(rec.match_quality).toBe("exact_version");
    expect(rec.image).toBe(`${QUAY_BIOCONTAINERS_PREFIX}/${protoRepo}:${targetHash}-0`);
    expect(asked).toEqual([repoUrl(protoRepo), repoUrl("__proto__")]);
  });

  it("falls back to a name-only newest tag when no tried combination is built", async () => {
    const asked = stubQuay({
      [repo]: ["unbuiltcombohash-0"],
      bwa: ["0.7.18--h0_0", "0.7.17--h0_0"],
    });
    const rec = await recommendContainer([{ name: "samtools", version: "1.17" }, { name: "bwa" }]);
    expect(rec.match_quality).toBe("name_only");
    expect(rec.image).toBe(`${QUAY_BIOCONTAINERS_PREFIX}/${repo}:unbuiltcombohash-0`);
    expect(rec.notes).toEqual([
      "could not resolve a built version combination; using newest available build",
    ]);
    expect(asked).toEqual([repoUrl(repo), repoUrl("bwa")]);
  });

  it("validates a pinned version inside resolution, where Python validates it", async () => {
    // A partly pinned set builds no targets until _resolve_versions does, so Python probes
    // the repo and the unpinned package's tags first and only then refuses. Same here.
    const asked = stubQuay({ [repo]: ["somehash-0"], bwa: ["0.7.17--h0_0"] });
    await expect(
      recommendContainer([{ name: "samtools", version: "1 2" }, { name: "bwa" }]),
    ).rejects.toThrow("Invalid version [1 2] encountered.");
    expect(asked).toEqual([repoUrl(repo), repoUrl("bwa")]);
  });
});

// ---------------------------------------------------------------------------
// recommendContainer, the request itself -- test_recommend.py
// ---------------------------------------------------------------------------

describe("recommendContainer (the request)", () => {
  it("has nothing to recommend for no packages, and asks quay.io nothing", async () => {
    const calls = stubNoNetwork();
    const rec = await recommendContainer([]);
    expect(rec.image).toBeNull();
    expect(rec.source).toBe("none");
    expect(rec.notes).toEqual(["no packages supplied"]);
    expect(calls.count).toBe(0);
  });

  it("drops an entry with no name at all", async () => {
    const calls = stubNoNetwork();
    const rec = await recommendContainer([{ name: "" }]);
    expect(rec.image).toBeNull();
    expect(rec.notes).toEqual(["no packages supplied"]);
    expect(calls.count).toBe(0);
  });

  it("answers the second identical request from the cache", async () => {
    const asked = stubQuay({ samtools: ["1.17--h0_0"] });
    const first = await recommendContainer(single("samtools", "1.17"));
    const second = await recommendContainer(single("samtools", "1.17"));
    expect(asked).toHaveLength(1);
    expect(second).toEqual(first);
  });

  it("keys the cache on the whole (name, version) pairs, sorted", async () => {
    // Two entries with the same name differ only by version, so a key sorted on names
    // alone maps a request and its reverse to two entries -- two lookups, and two
    // different version hashes, for what is the same set of packages.
    const dupRepo = v2RepoName(["bwa", "bwa"]);
    const forwards: PackageSpec[] = [
      { name: "bwa", version: "0.7.17" },
      { name: "bwa", version: "0.7.18" },
    ];
    const backwards: PackageSpec[] = [...forwards].reverse();
    expect(v2VersionHash(forwards)).not.toBe(v2VersionHash(backwards));

    const asked = stubQuay({ [dupRepo]: ["somehash-0"] });
    const first = await recommendContainer(forwards);
    const second = await recommendContainer(backwards);
    expect(asked).toEqual([repoUrl(dupRepo)]);
    expect(second).toEqual(first);
  });

  it("does not let the cache answer a request made with different options", async () => {
    const repo = v2RepoName(["bwa", "samtools"]);
    const asked = stubQuay({ [repo]: ["somehash-0"], bwa: ["0.7.17--h0_0"] });
    const specs: PackageSpec[] = [{ name: "samtools", version: "1.17" }, { name: "bwa" }];
    await recommendContainer(specs, { resolveVersions: false });
    await recommendContainer(specs, { resolveVersions: true });
    expect(asked).toEqual([repoUrl(repo), repoUrl(repo), repoUrl("bwa")]);
  });

  it("goes back to quay.io when the cache is bypassed", async () => {
    const asked = stubQuay({ samtools: ["1.17--h0_0"] });
    await recommendContainer(single("samtools", "1.17"), { useCache: false });
    await recommendContainer(single("samtools", "1.17"), { useCache: false });
    expect(asked).toHaveLength(2);
  });

  it("expires a cached entry on the monotonic clock after 300s", async () => {
    const asked = stubQuay({ samtools: ["1.17--h0_0"] });
    const now = vi.spyOn(performance, "now").mockReturnValue(1_000);
    await recommendContainer(single("samtools", "1.17"));
    expect(asked).toHaveLength(1);
    now.mockReturnValue(1_000 + 299_999);
    await recommendContainer(single("samtools", "1.17"));
    expect(asked).toHaveLength(1);
    now.mockReturnValue(1_000 + 300_000);
    await recommendContainer(single("samtools", "1.17"));
    expect(asked).toHaveLength(2);
  });

  it("does not let the wall clock going backwards extend a cached entry", async () => {
    // Python reads time.monotonic(), which an NTP correction cannot move; Date.now() it
    // can, and a backwards step would keep a negative recommendation alive past its TTL.
    const asked = stubQuay({ samtools: ["1.17--h0_0"] });
    const now = vi.spyOn(performance, "now").mockReturnValue(1_000);
    const wall = vi.spyOn(Date, "now").mockReturnValue(1_800_000_000_000);
    await recommendContainer(single("samtools", "1.17"));
    expect(asked).toHaveLength(1);
    // An hour of wall clock lost while five minutes of real time pass.
    wall.mockReturnValue(1_800_000_000_000 - 3_600_000);
    now.mockReturnValue(1_000 + 300_001);
    await recommendContainer(single("samtools", "1.17"));
    expect(asked).toHaveLength(2);
  });

  it("picks the tag quay.io listed first when the sort cannot separate two", async () => {
    // Python's dict keeps the response order, so an unpinned samtools resolves to :1.0 there.
    // Object.keys would have put "1" first and recommended a different image.
    stubBody(() => okText('{"tags":{"1.0":{},"1":{}}}'));
    const rec = await recommendContainer(single("samtools"));
    expect(rec.image).toBe(`${QUAY_BIOCONTAINERS_PREFIX}/samtools:1.0`);
  });

  it("resolves through a 302 that carries a tag list", async () => {
    const calls = stubBody(() => statusWithBody(302, '{"tags":{"1.17--h0_0":{}}}'));
    const rec = await recommendContainer(single("samtools", "1.17"));
    expect(rec.image).toBe(`${QUAY_BIOCONTAINERS_PREFIX}/samtools:1.17--h0_0`);
    expect(rec.match_quality).toBe("exact_version");
    expect(calls.count).toBe(1);
  });

  it("strips Python's whitespace from a name and a version before looking anything up", async () => {
    // U+0085 either side of both, which str.strip() removes and String.prototype.trim keeps.
    const asked = stubQuay({ samtools: ["1.17--h0_0", "1.16--h0_0"] });
    const rec = await recommendContainer([
      { name: "\u0085SamTools\u0085", version: "\u00851.17\u0085" },
    ]);
    expect(asked).toEqual([repoUrl("samtools")]);
    expect(rec.image).toBe(`${QUAY_BIOCONTAINERS_PREFIX}/samtools:1.17--h0_0`);
    expect(rec.match_quality).toBe("exact_version");
  });

  it("keeps a U+FEFF in a version, which Python never finds a tag for", async () => {
    stubQuay({ samtools: ["1.17--h0_0"] });
    const rec = await recommendContainer([{ name: "samtools", version: "\ufeff1.17\ufeff" }]);
    expect(rec.match_quality).toBe("name_only");
    expect(rec.notes).toEqual([
      "requested version \ufeff1.17\ufeff not found; newest available tag is 1.17--h0_0",
    ]);
  });

  it("lets a malformed quay.io response through rather than calling it a failed lookup", async () => {
    stubBody(() => okText('{"namespace":"biocontainers"}'));
    await expect(recommendContainer(single("samtools", "1.17"))).rejects.toThrow(
      "no tags description found",
    );
  });
});

// ---------------------------------------------------------------------------
// biocontainerTagBuilt -- test_recommend.py
// ---------------------------------------------------------------------------

describe("biocontainerTagBuilt", () => {
  it("is true when the tag is among the repository's built tags", async () => {
    const asked = stubQuay({ pandas: ["2.2.1", "2.1.4--py311h1128e8f_0", "2.0.0"] });
    expect(await biocontainerTagBuilt("quay.io/biocontainers/pandas:2.2.1")).toBe(true);
    expect(asked).toEqual([repoUrl("pandas")]);
  });

  it("is false when the repository has tags and this is not one of them", async () => {
    const asked = stubQuay({ pandas: ["2.2.1", "2.2.0", "2.1.4"] });
    expect(await biocontainerTagBuilt("quay.io/biocontainers/pandas:2.1.4--py311h1128e8f_0")).toBe(
      false,
    );
    expect(asked).toEqual([repoUrl("pandas")]);
  });

  it("is null for an empty repository, which cannot tell absent from blipping", async () => {
    const asked = stubQuay({});
    expect(
      await biocontainerTagBuilt("quay.io/biocontainers/pandas:2.1.4--py311h1128e8f_0"),
    ).toBeNull();
    expect(asked).toEqual([repoUrl("pandas")]);
  });

  it("is null on an HTTP error", async () => {
    const calls = stubBody(() => status(500));
    expect(await biocontainerTagBuilt("quay.io/biocontainers/pandas:2.2.1")).toBeNull();
    expect(calls.count).toBe(1);
  });

  it("is null on a connection error", async () => {
    const calls = stubNoNetwork();
    expect(await biocontainerTagBuilt("quay.io/biocontainers/pandas:2.2.1")).toBeNull();
    expect(calls.count).toBe(1);
  });

  it.each([
    ["ubuntu:latest", "not a biocontainers reference"],
    ["quay.io/biocontainers/pandas", "no tag"],
    ["quay.io/biocontainers/:2.2.1", "no repository"],
    ["quay.io/biocontainers/pandas:", "an empty tag"],
    ["", "empty"],
    ["docker.io/library/python:3.13", "a different registry"],
  ])("is null for %s (%s), without consulting the network", async (image) => {
    const calls = stubNoNetwork();
    expect(await biocontainerTagBuilt(image)).toBeNull();
    expect(calls.count).toBe(0);
  });

  it("verifies a mulled-v2 reference, whose repo name carries no colon", async () => {
    const repo = v2RepoName(["bwa", "samtools"]);
    stubQuay({ [repo]: ["abc-0"] });
    expect(await biocontainerTagBuilt(`${QUAY_BIOCONTAINERS_PREFIX}/${repo}:abc-0`)).toBe(true);
    expect(await biocontainerTagBuilt(`${QUAY_BIOCONTAINERS_PREFIX}/${repo}:abc-9`)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// The build number, which is a Python `\d+$` search and not a JavaScript one
// ---------------------------------------------------------------------------

describe("parse_tag's build number", () => {
  it("reads a build number written in Nd digits, as Python's `\\d` does", () => {
    // Python's answer: version_sorted(["1.0--a_٢", "1.0--b_1"]) is
    // ["1.0--a_٢", "1.0--b_1"], because ARABIC-INDIC DIGIT TWO is `\d` there and the build
    // number is 2. A JavaScript `\d` finds no build number at all, which makes it -1 and puts
    // the tag below b_1.
    expect(versionSorted(["1.0--a_٢", "1.0--b_1"])).toEqual([
      "1.0--a_٢",
      "1.0--b_1",
    ]);
  });

  it("lets `$` match before a trailing newline, as a Python `$` does", () => {
    // A tag name is a JSON object key, so it can carry a newline. Python's `$` matches just
    // before one at the end of the string, so "1.0--a_2\n" has build number 2 and sorts above
    // "1.0--a_1"; a JavaScript `$` without the `m` flag matches only at the very end, which
    // left it at -1 and reversed the pair.
    expect(versionSorted(["1.0--a_2\n", "1.0--a_1"])).toEqual(["1.0--a_2\n", "1.0--a_1"]);
    // But not before a newline that is not the last character, which Python will not either.
    expect(versionSorted(["1.0--a_2\n\n", "1.0--a_1"])).toEqual(["1.0--a_1", "1.0--a_2\n\n"]);
  });

  it("reads a build number past 2^53 exactly", () => {
    expect(versionSorted(["1.0--h_9007199254740993", "1.0--h_9007199254740992"])[0]).toBe(
      "1.0--h_9007199254740993",
    );
  });
});

// ---------------------------------------------------------------------------
// The response charset, which requests decides and this used to ignore
// ---------------------------------------------------------------------------

/** UTF-16LE bytes, so a fixture can arrive in something other than UTF-8. */
function utf16le(text: string): Uint8Array {
  const out = new Uint8Array(text.length * 2);
  for (let i = 0; i < text.length; i += 1) {
    const unit = text.charCodeAt(i);
    out[i * 2] = unit & 0xff;
    out[i * 2 + 1] = unit >> 8;
  }
  return out;
}

/** UTF-32LE bytes, which the Encoding Standard has no decoder for at all. */
function utf32le(text: string): Uint8Array {
  const points = [...text];
  const out = new Uint8Array(points.length * 4);
  points.forEach((ch, i) => {
    const cp = ch.codePointAt(0)!;
    out[i * 4] = cp & 0xff;
    out[i * 4 + 1] = (cp >> 8) & 0xff;
    out[i * 4 + 2] = (cp >> 16) & 0xff;
    out[i * 4 + 3] = (cp >> 24) & 0xff;
  });
  return out;
}

const latin1 = (text: string) => Uint8Array.from([...text].map((c) => c.codePointAt(0)!));

/** One response with these bytes and this content type, for every repository asked for. */
function stubBytes(body: Uint8Array, contentType?: string): { count: number } {
  const calls = { count: 0 };
  vi.stubGlobal("fetch", async () => {
    calls.count++;
    return new Response(body, {
      status: 200,
      headers: contentType === undefined ? {} : { "content-type": contentType },
    });
  });
  return calls;
}

describe("encodingFromContentType", () => {
  it.each([
    // Every expected value below is what requests 2.34.2's get_encoding_from_headers returned
    // for that header in the reference checkout's interpreter.
    [null, null],
    ["", null],
    ["application/json", "utf-8"],
    ["application/json; charset=utf-16le", "utf-16le"],
    ["application/json;charset='UTF-16LE'", "UTF-16LE"],
    ['text/plain; charset="utf-8"', "utf-8"],
    ["text/plain", "ISO-8859-1"],
    // 2.34.2 stopped lower-casing the media type, so these match neither test.
    ["TEXT/PLAIN", null],
    ["APPLICATION/JSON", null],
    ["application/octet-stream", null],
    // An empty charset is "no encoding" to requests, and is carried as itself.
    ["application/json; charset=", ""],
    // A parameter with no "=" is not a charset at all, so the media type still decides.
    ["application/json; charset", "utf-8"],
    ["application/json; CHARSET=Latin-1", "Latin-1"],
    ["  application/json  ; charset =  utf-8 ", "utf-8"],
  ])("reads %j as %j", (header, expected) => {
    expect(encodingFromContentType(header)).toBe(expected);
  });

  it("takes the last of a repeated charset, as the dict requests builds does", () => {
    expect(encodingFromContentType("application/json; charset=utf-8; charset=utf-16le")).toBe(
      "utf-16le",
    );
  });
});

describe("requestsResponseJson", () => {
  const body = '{"tags":{"1.17--h0_0":{}}}';

  it("decodes by the declared charset", () => {
    const out = requestsResponseJson(utf16le(body), "application/json; charset=utf-16le");
    expect(out.text).toBe(body);
    expect(out.value).toEqual({ tags: { "1.17--h0_0": {} } });
  });

  it("sniffs the UTF when nothing is declared, as guess_json_utf does", () => {
    expect(requestsResponseJson(utf16le(body), null).text).toBe(body);
    expect(requestsResponseJson(utf32le(body), null).text).toBe(body);
    // A BOM is enough on its own, and utf-8-sig removes it.
    const bom = Uint8Array.from([0xef, 0xbb, 0xbf, ...new TextEncoder().encode(body)]);
    expect(requestsResponseJson(bom, null).text).toBe(body);
    const utf16Bom = Uint8Array.from([0xff, 0xfe, ...utf16le(body)]);
    expect(requestsResponseJson(utf16Bom, null).text).toBe(body);
  });

  it("does not sniff once the headers have said something, even a wrong something", () => {
    // Python: encoding is "utf-8" from the media type, so json() skips guess_json_utf and
    // parses the UTF-16 body decoded as UTF-8 -- which fails.
    expect(() => requestsResponseJson(utf16le(body), "application/json")).toThrow();
  });

  it("keeps a BOM that the declared codec keeps, so the parse fails as Python's does", () => {
    // Python's "utf-8" codec decodes a BOM to U+FEFF and json.loads refuses it; TextDecoder
    // would have swallowed it and parsed happily.
    const bom = Uint8Array.from([0xef, 0xbb, 0xbf, ...new TextEncoder().encode(body)]);
    expect(() => requestsResponseJson(bom, "application/json; charset=utf-8")).toThrow();
  });

  it("decodes latin-1 as Python does, not as windows-1252", () => {
    // 0x80 is U+0080 in Python's ISO-8859-1 and U+20AC in the Encoding Standard's, which maps
    // that label to windows-1252. A text/* media type with no charset is latin-1 to requests.
    const tagged = '{"tags":{"1.0--h_0":{}}}';
    const out = requestsResponseJson(latin1(tagged), "text/plain");
    expect(out.text).toBe(tagged);
    expect(new TextDecoder("iso-8859-1").decode(latin1(tagged))).not.toBe(tagged);
  });

  it("falls through to the text when the sniffed codec cannot decode the bytes", () => {
    // No nulls, so guess_json_utf says utf-8; the bytes are not utf-8, so the strict decode
    // fails and requests re-reads the body leniently. A SyntaxError and not a TypeError is the
    // point: the strict decode's failure has to come back as the UnicodeDecodeError it stands
    // for and be fallen through, not escape as whatever TextDecoder threw.
    expect(() => requestsResponseJson(Uint8Array.from([0x80, 0x81, 0x82, 0x83]), null)).toThrow(
      SyntaxError,
    );
  });

  it("gives an empty body the empty text, as Response.text does", () => {
    // json.loads("") is a JSONDecodeError there and JSON.parse("") a SyntaxError here, which
    // quayTagsFor turns into the same failed lookup.
    expect(() => requestsResponseJson(new Uint8Array(0), "application/json")).toThrow(SyntaxError);
  });

  it("refuses an unknown charset the way a LookupError is refused, not by throwing it", () => {
    // Python: str(content, "cp9999", errors="replace") raises LookupError, which Response.text
    // catches and re-decodes as utf-8. The body is valid JSON either way.
    const out = requestsResponseJson(
      new TextEncoder().encode(body),
      "application/json; charset=cp9999",
    );
    expect(out.value).toEqual({ tags: { "1.17--h0_0": {} } });
  });
});

describe("quayTagsFor (the charset)", () => {
  it("reads a UTF-16LE tag list, which was a cached lookup failure before", async () => {
    stubBytes(utf16le(quayRepo(["1.17--h0_0", "1.16--h0_0"])), "application/json; charset=utf-16le");
    expect(await quayTagsFor("samtools")).toEqual(["1.17--h0_0", "1.16--h0_0"]);
  });

  it("keeps the response order through a non-UTF-8 decode", async () => {
    stubBytes(utf16le('{"tags":{"1.0":{},"1":{}}}'), "application/json; charset=utf-16le");
    expect(await quayTagsFor("samtools")).toEqual(["1.0", "1"]);
  });
});

describe("recommendContainer (the charset)", () => {
  it("resolves samtools=1.17 from a UTF-16LE response", async () => {
    // Python resolves this and the port returned "lookup failed", then cached
    // that failure for five minutes.
    stubBytes(utf16le(quayRepo(["1.17--h0_0"])), "application/json; charset=utf-16le");
    const rec = await recommendContainer(single("samtools", "1.17"));
    expect(rec.image).toBe(`${QUAY_BIOCONTAINERS_PREFIX}/samtools:1.17--h0_0`);
    expect(rec.match_quality).toBe("exact_version");
    expect(rec.notes).toEqual([]);
  });

  it("resolves it from a sniffed UTF-32LE response with no content type at all", async () => {
    stubBytes(utf32le(quayRepo(["1.17--h0_0"])));
    const rec = await recommendContainer(single("samtools", "1.17"));
    expect(rec.image).toBe(`${QUAY_BIOCONTAINERS_PREFIX}/samtools:1.17--h0_0`);
  });

  it("replaces a bad UTF-32 unit once, as Python's codec does, and still reads the JSON", async () => {
    // An earlier revision of the accepted-divergence list waved this away with "a body that is
    // not valid UTF-32 is not JSON", which is false: four bytes standing for a surrogate become
    // one U+FFFD inside a string literal and the body parses. With the charset declared,
    // `Response.json` reads `self.text`, which decodes with errors="replace" and gives Python the
    // same `1.0--\uFFFD`; with no header at all `guess_json_utf` answers utf-32-le, the strict
    // decode raises at bytes 60-63 and the fallback to `Response.text` lands in the same place.
    // The count is one U+FFFD per four-byte unit on both, which is what `decodeUtf32` is for.
    const body = utf32le('{"tags":{"1.0--X":{}}}');
    const at = body.indexOf(0x58);
    body.set([0x00, 0xd8, 0x00, 0x00], at);
    stubBytes(body, "application/json; charset=utf-32-le");
    expect(await quayTagsFor("samtools")).toEqual(["1.0--\uFFFD"]);
  });
});

// ---------------------------------------------------------------------------
// Names Python will not encode, which it refuses before any request
// ---------------------------------------------------------------------------

describe("an ill-formed package name or version", () => {
  it("is refused at the mulled-v2 name hash, as `buffer.encode()` is", async () => {
    // Python raises UnicodeEncodeError inside v2_image_name, before a request;
    // Node substitutes U+FFFD and hashes a different set of names, so quay.io gets asked about
    // a different repository and answers about it confidently.
    const calls = stubNoNetwork();
    const err = await recommendContainer([
      { name: "a\ud800", version: "1" },
      { name: "b", version: "1" },
    ]).then(
      () => null,
      (e: unknown) => e,
    );
    expect(err).toBeInstanceOf(GalaxyValidationError);
    expect((err as Error).message).toBe(
      "'utf-8' codec can't encode character '\\ud800' in position 1: surrogates not allowed",
    );
    expect(calls.count).toBe(0);
  });

  it("is refused at the version hash too, which Python encodes second", async () => {
    const calls = stubNoNetwork();
    const err = await recommendContainer([
      { name: "a", version: "1\ud800" },
      { name: "b", version: "1" },
    ]).then(
      () => null,
      (e: unknown) => e,
    );
    expect(err).toBeInstanceOf(GalaxyValidationError);
    expect((err as Error).message).toContain("in position 1:");
    expect(calls.count).toBe(0);
  });

  it("is NOT refused at the URL, where Python encodes it with surrogatepass", async () => {
    // The one place Python does not refuse a surrogate: urllib3 encodes a URL component with
    // errors="surrogatepass", so `requests` asks quay.io about `a%ED%A0%80` rather than raising.
    // A single package never reaches a hash, so this is a lookup in Python -- and it was a
    // refusal here until the URL was prepared the way requests prepares it.
    // `requests.Request("GET", url).prepare().url` on 2.34.2 is the oracle.
    const asked = stubQuay({});
    const rec = await recommendContainer(single("a\ud800", "1"));
    expect(rec.found).toBe(false);
    expect(asked).toEqual([repoUrl("a%ED%A0%80")]);
  });

  it("leaves a well-formed astral name alone, because Python encodes that", async () => {
    const asked = stubQuay({});
    const rec = await recommendContainer(single("a\u{10000}", "1"));
    expect(rec.found).toBe(false);
    // `quote()` percent-encodes the utf-8 bytes; the raw character is not what goes on the wire.
    expect(asked).toEqual([repoUrl("a%F0%90%80%80")]);
  });
});

// ---------------------------------------------------------------------------
// The URL requests would have sent, which is not the one fetch parses
// ---------------------------------------------------------------------------

describe("the repository URL", () => {
  it("asks about the backslash repository requests asks about", async () => {
    // `a\b` is `/biocontainers/a%5Cb` to requests and `/biocontainers/a/b` to fetch's URL
    // parser, which treats a backslash in an https URL as a path separator. With the escaped
    // repository holding a tag and the split path a 404, Python finds the image and this used
    // to report no container at all.
    const asked: string[] = [];
    vi.stubGlobal("fetch", async (url: string) => {
      const requested = asRequested(url);
      asked.push(requested);
      return requested === repoUrl("a%5Cb") ? okText(quayRepo(["1.17--h0_0"])) : status(404);
    });
    const rec = await recommendContainer(single("a\\b", "1.17"));
    expect(asked).toEqual([repoUrl("a%5Cb")]);
    expect(rec.image).toBe(`${QUAY_BIOCONTAINERS_PREFIX}/a\\b:1.17--h0_0`);
    expect(rec.match_quality).toBe("exact_version");
  });

  // Every expected value is `requests.Request("GET", url).prepare().url` on the installed
  // 2.34.2, compared after a `new URL()` round trip so it is the request and not the argument.
  it.each([
    ["a\\b", "a%5Cb"],
    ["a|b", "a%7Cb"],
    ["a^b", "a%5Eb"],
    ["a[b]", "a%5Bb%5D"],
    ["a{b}", "a%7Bb%7D"],
    ["a<b>c", "a%3Cb%3Ec"],
    ["a`b", "a%60b"],
    ["a\u0001b", "a%01b"],
    // Interior whitespace survives normalize's strip and reaches here without build_target,
    // which only the pinned paths call -- and fetch's parser DELETES a tab or a newline.
    ["a b", "a%20b"],
    ["a\tb", "a%09b"],
    ["café", "caf%C3%A9"],
    // A valid escape survives, uppercased -- and one standing for an unreserved character is
    // decoded, because requote_uri unquotes those before it quotes anything.
    ["a%5cb", "a%5Cb"],
    ["a%2fb", "a%2Fb"],
    ["a%41b", "aAb"],
    ["a%7eb", "a~b"],
    ["a%252fb", "a%252fb"],
    // A `%` that begins no escape makes every `%` in the component an escaped one.
    ["a%zzb", "a%25zzb"],
    ["a%b", "a%25b"],
    ["a%41%zz", "a%2541%25zz"],
    // The sub-delimiters and `:@/` are left alone, as urllib3 leaves them.
    ["a+b", "a+b"],
    ["a,b", "a,b"],
    ["a;b", "a;b"],
    ["a:b", "a:b"],
    ["a@b", "a@b"],
    ["a~b", "a~b"],
    // A `?` or `#` in the name is still a delimiter to requests, which splits the query and the
    // fragment off before it encodes anything -- so neither is escaped here either.
    ["a?b", "a?b"],
    ["a#b", "a#b"],
    ["a?b|c", "a?b%7Cc"],
  ])("prepares %j the way requests prepares it", async (repo, prepared) => {
    const asked = stubQuay({});
    await quayTagsFor(repo);
    expect(asked).toEqual([repoUrl(prepared)]);
  });

  // urllib3 removes the path's dot segments BEFORE anything is unquoted and over the path of the
  // whole URL, so a literal `..` climbs out of `/biocontainers/` and a `%2e` is not a dot yet.
  // Every expected value is `requests.Request("GET", url).prepare().url` on 2.34.2.
  it.each([
    // The r6 finding: the `..` pops the `%2e`, not the `a`, so the repository is `a/b`.
    ["a/%2e/../b", `${API}/biocontainers/a/b`],
    ["a/%2E/../b", `${API}/biocontainers/a/b`],
    ["a/./%2e/../b", `${API}/biocontainers/a/b`],
    ["a/./b", `${API}/biocontainers/a/b`],
    ["a/../b", `${API}/biocontainers/b`],
    ["a/b/../c", `${API}/biocontainers/a/c`],
    [".", `${API}/biocontainers/`],
    ["..", `${API}/`],
    ["./a", `${API}/biocontainers/a`],
    ["../a", `${API}/a`],
    ["a/.", `${API}/biocontainers/a/`],
    ["a/..", `${API}/biocontainers/`],
    ["a/../../b", `${API}/b`],
    ["../../../..", "https://quay.io/"],
    // Three dots is a segment like any other, and an escaped separator is not one.
    ["...", `${API}/biocontainers/...`],
    ["a/%2f../b", `${API}/biocontainers/a/%2F../b`],
    ["a/..%2fb", `${API}/biocontainers/a/..%2Fb`],
  ])("removes the dot segments from %j where urllib3 removes them", async (repo, prepared) => {
    const asked: string[] = [];
    vi.stubGlobal("fetch", async (url: string) => {
      asked.push(asRequested(url));
      return okText(quayRepo([]));
    });
    await quayTagsFor(repo);
    expect(asked).toEqual([prepared]);
  });

  // Accepted divergence 8: the escape requote_uri decodes back into a dot segment. The URL this
  // builds is Python's to the character -- the first column -- and the URL Standard then
  // collapses the segment on the way to the wire, where `requests` sends it as prepared.
  it.each([
    ["%2e", `${API}/biocontainers/.`, `${API}/biocontainers/`],
    ["%2E", `${API}/biocontainers/.`, `${API}/biocontainers/`],
    ["%2e%2e", `${API}/biocontainers/..`, `${API}/`],
    ["%2e/a", `${API}/biocontainers/./a`, `${API}/biocontainers/a`],
    ["a/%2e/b", `${API}/biocontainers/a/./b`, `${API}/biocontainers/a/b`],
    ["a/%2e%2e/b", `${API}/biocontainers/a/../b`, `${API}/biocontainers/b`],
  ])("prepares %j as requests does and loses it to fetch's parse", async (repo, prepared, sent) => {
    const built: string[] = [];
    const asked: string[] = [];
    vi.stubGlobal("fetch", async (url: string) => {
      built.push(url);
      asked.push(asRequested(url));
      return okText(quayRepo([]));
    });
    await quayTagsFor(repo);
    expect(built).toEqual([prepared]);
    expect(asked).toEqual([sent]);
  });

  it("names the prepared URL in the error, as requests' HTTPError does", async () => {
    stubBody(() => status(404));
    const err = await quayTagsFor("a\\b").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(QuayHttpStatusError);
    expect((err as Error).message).toBe(`404 Error for url: ${repoUrl("a%5Cb")}`);
  });
});

// ---------------------------------------------------------------------------
// json.loads is Python's JSON, not RFC 8259's
// ---------------------------------------------------------------------------

describe("a body carrying NaN or Infinity", () => {
  it.each(["NaN", "Infinity", "-Infinity"])(
    "reads the tag list through a %s value, as json.loads does",
    async (constant) => {
      // Galaxy reads the tag NAMES and ignores the metadata, so a 200 whose metadata carries
      // one of these resolves in Python; JSON.parse refuses all three, which made it a failed
      // lookup cached for five minutes.
      stubBody(() => okText(`{"tags":{"1.17--h0_0":${constant}}}`));
      const rec = await recommendContainer(single("samtools", "1.17"));
      expect(rec.image).toBe(`${QUAY_BIOCONTAINERS_PREFIX}/samtools:1.17--h0_0`);
      expect(rec.match_quality).toBe("exact_version");
    },
  );

  it("accepts one inside an array and after whitespace, where a value may stand", async () => {
    stubBody(() => okText('{"tags":{"1.17--h0_0":[ NaN, -Infinity ]}}'));
    expect(await quayTagsFor("samtools")).toEqual(["1.17--h0_0"]);
  });

  it("keeps a tag NAMED NaN a tag name", async () => {
    // The three are values, not a token the scan may rewrite anywhere: inside a string literal
    // `NaN` is the name of a tag, and quay.io could publish one.
    stubBody(() => okText('{"tags":{"NaN":{}}}'));
    expect(await quayTagsFor("samtools")).toEqual(["NaN"]);
  });

  it.each(['{"tags":{NaN:{}}}', '{"tags":{"1.17":1 NaN}}', '{"tags":NaNx}'])(
    "still refuses %j, which json.loads refuses too",
    async (body) => {
      stubBody(() => okText(body));
      await expect(quayTagsFor("samtools")).rejects.toBeInstanceOf(QuayRequestError);
    },
  );
});

// ---------------------------------------------------------------------------
// int()'s digit limit, inside the JSON decoder
// ---------------------------------------------------------------------------

describe("a body whose integer literal is longer than int() will read", () => {
  // Python's decoder reads an integer with `int()`, so the limit is inside `json.loads` too:
  // the body is a ValueError there, and `JSON.parse` read it as `Infinity` and recommended the
  // tag. `Response.json` re-raises only a JSONDecodeError, so this one is not a failed lookup --
  // it leaves the recommender the way a long build number does. Every row is `json.loads` on the
  // reference interpreter.
  const nines = (n: number) => "9".repeat(n);
  const body = (value: string) => `{"tags":{"1.17--h0_0":${value}}}`;

  it("refuses 4301 digits in a value, as json.loads does", async () => {
    stubBody(() => okText(body(nines(4301))));
    const err = await recommendContainer(single("samtools", "1.17")).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(GalaxyValidationError);
    expect((err as Error).message).toBe(
      "Exceeds the limit (4300 digits) for integer string conversion: value has 4301 digits; " +
        "use sys.set_int_max_str_digits() to increase the limit",
    );
  });

  it.each([
    ["exactly 4300 digits", nines(4300)],
    // A `.` or an exponent makes it float(), which has no limit and is `inf` in both runtimes.
    ["a float that long", `${nines(4301)}.0`],
    ["an exponent that long", `${nines(4301)}e1`],
    // Inside a string literal it is not a number at all.
    ["the digits inside a string", `"${nines(4301)}"`],
  ])("reads %s, as json.loads does", async (_what, value) => {
    stubBody(() => okText(body(value)));
    const rec = await recommendContainer(single("samtools", "1.17"));
    expect(rec.image).toBe(`${QUAY_BIOCONTAINERS_PREFIX}/samtools:1.17--h0_0`);
    expect(rec.match_quality).toBe("exact_version");
  });

  it.each([
    ["the sign, which is not a digit", body(`-${nines(4301)}`)],
    ["one nested in an array", body(`[1, ${nines(4301)}]`)],
    // A left-to-right pass reaches the integer before it notices either of these.
    ["a syntax error after it", body(`${nines(4301)} x`)],
    ["a body that stops after it", `{"tags":{"1.17--h0_0":${nines(4301)}`],
    ["a NaN before it", `{"tags":{"1.17--h0_0":NaN},"x":${nines(4301)}}`],
  ])("refuses it through %s", async (_what, text) => {
    stubBody(() => okText(text));
    const err = await quayTagsFor("samtools").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(GalaxyValidationError);
    expect((err as Error).message).toContain("Exceeds the limit (4300 digits)");
  });

  it.each([
    // The decoder reports the delimiter it wanted at column 25 and never reads the integer.
    ["a syntax error before it", body(`1 ${nines(4301)}`)],
    ["standing where a key belongs", `{"tags":{${nines(4301)}:1}}`],
  ])("is a failed lookup for %s, which json.loads refuses first", async (_what, text) => {
    stubBody(() => okText(text));
    const err = await quayTagsFor("samtools").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(QuayRequestError);
    expect(err).not.toBeInstanceOf(GalaxyValidationError);
  });
});

// ---------------------------------------------------------------------------
// int()'s digit limit, which a build number can cross
// ---------------------------------------------------------------------------

describe("a tag whose build number is longer than int() will read", () => {
  it("refuses it the way parse_tag's ValueError reaches the caller", async () => {
    // int() stops at sys.get_int_max_str_digits() digits, so parse_tag raises inside
    // version_sorted; recommend_container's `except RequestException` does not catch a
    // ValueError, so it leaves the recommender and the tool re-raises it.
    stubBody(() => okText(quayRepo([`1.0--a_${"9".repeat(4301)}`])));
    const err = await recommendContainer(single("samtools")).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(GalaxyValidationError);
    expect((err as Error).message).toContain("Exceeds the limit (4300 digits)");
  });

  it("reads one of exactly 4300 digits, as int() does", async () => {
    const tag = `1.0--a_${"9".repeat(4300)}`;
    stubBody(() => okText(quayRepo([tag])));
    const rec = await recommendContainer(single("samtools"));
    expect(rec.image).toBe(`${QUAY_BIOCONTAINERS_PREFIX}/samtools:${tag}`);
  });

  it("refuses one in the VERSION half too, where packaging's int() is", async () => {
    // The build number is not the only number in a tag: `parse_version` calls `int()` on the
    // release, the epoch, a pre / post / dev numeral and a numeric local part, and the same
    // ValueError comes out of any of them -- so a tag of 4301 nines is refused as well.
    stubBody(() => okText(quayRepo([`${"9".repeat(4301)}--h0_0`])));
    const err = await recommendContainer(single("samtools")).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(GalaxyValidationError);
    expect((err as Error).message).toContain("Exceeds the limit (4300 digits)");
  });
});

// ---------------------------------------------------------------------------
// The note's name list, which Python formats with repr
// ---------------------------------------------------------------------------

describe("the no-multi-package note", () => {
  it("escapes a backslash in a name, as repr does", async () => {
    // Python's note is "... for ['a\\b', 'c']" and interpolating the names
    // directly gave "... for ['a\b', 'c']" -- a different string in the returned contract.
    stubQuay({});
    const rec = await recommendContainer([
      { name: "a\\b", version: "1" },
      { name: "c", version: "1" },
    ]);
    expect(rec.notes).toEqual(["no multi-package biocontainer built for ['a\\\\b', 'c']"]);
  });

  it("escapes a non-printable code point, as repr does", async () => {
    stubQuay({});
    const rec = await recommendContainer([
      { name: "a\u200b", version: "1" },
      { name: "c", version: "1" },
    ]);
    expect(rec.notes).toEqual(["no multi-package biocontainer built for ['a\\u200b', 'c']"]);
  });

  it("switches to double quotes for a name holding a single one", async () => {
    // build_target refuses a quote in a package name, so this goes through pyReprList directly
    // in the python-str tests; here the plain case pins that nothing else changed.
    stubQuay({});
    const rec = await recommendContainer([
      { name: "bwa", version: "1" },
      { name: "samtools", version: "1" },
    ]);
    expect(rec.notes).toEqual(["no multi-package biocontainer built for ['bwa', 'samtools']"]);
  });
});
