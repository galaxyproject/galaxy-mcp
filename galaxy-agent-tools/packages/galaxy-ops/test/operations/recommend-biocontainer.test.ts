/**
 * The op around the recommender: the `name=version` parsing, the shape Python's tool
 * returns, and the `verified` round trip. The recommender's own behaviour is pinned
 * against Galaxy's unit tests in `test/mulled.test.ts`; this file is about the surface.
 *
 * No network: `fetch` is stubbed for every case.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  recommendBiocontainerOp,
  recommendBiocontainer,
  parsePackageSpecs,
} from "../../src/operations/recommend-biocontainer";
import { __clearRecommendationCacheForTest, v2RepoName, v2VersionHash } from "../../src/mulled";
import { GalaxyValidationError } from "../../src/errors";
import { DEFAULT_POLL, type GalaxyContext } from "../../src/context";

const ctx = { client: undefined as never, poll: DEFAULT_POLL } as unknown as GalaxyContext;

/**
 * A quay.io repository body with these tags, as `mulled.test.ts` builds one: text rather than
 * an object, because `JSON.stringify` would reorder an integer-shaped tag name.
 */
const quayRepo = (tags: readonly string[]) =>
  `{"namespace":"biocontainers","tags":{${tags
    .map((tag) => `${JSON.stringify(tag)}:{"name":${JSON.stringify(tag)},"size":1024}`)
    .join(",")}}}`;

/** A real `Response`: the recommender streams the body, so a `{ json() }` stand-in is not one. */
const okText = (text: string) =>
  new Response(text, { status: 200, headers: { "content-type": "application/json" } });

function stubQuay(byRepo: Record<string, readonly string[]>, asked: string[] = []): string[] {
  vi.stubGlobal("fetch", async (url: string) => {
    asked.push(url);
    const repo = url.slice(url.lastIndexOf("/") + 1);
    return okText(quayRepo(byRepo[repo] ?? []));
  });
  return asked;
}

beforeEach(() => {
  __clearRecommendationCacheForTest();
});

afterEach(() => {
  vi.unstubAllGlobals();
  __clearRecommendationCacheForTest();
});

describe("parsePackageSpecs", () => {
  it("reads 'name' and 'name=version', trimming both", () => {
    expect(parsePackageSpecs([" samtools = 1.17 ", "bwa"])).toEqual([
      { name: "samtools", version: "1.17" },
      { name: "bwa", version: null },
    ]);
  });

  it("strips Python's whitespace, not JavaScript's", () => {
    // str.strip() removes U+0085 and U+001C-U+001F, which trim keeps, and keeps U+FEFF, which
    // trim removes. Both directions change which tag this resolves to.
    expect(parsePackageSpecs(["\u0085samtools\u001c=\u00851.17\u0085"])).toEqual([
      { name: "samtools", version: "1.17" },
    ]);
    expect(parsePackageSpecs(["samtools=\ufeff1.17\ufeff"])).toEqual([
      { name: "samtools", version: "\ufeff1.17\ufeff" },
    ]);
  });

  it("treats 'name=' as unpinned, as Python's partition does", () => {
    expect(parsePackageSpecs(["samtools="])).toEqual([{ name: "samtools", version: null }]);
  });

  it("splits on the FIRST '=' so a version may contain one", () => {
    expect(parsePackageSpecs(["r-ggplot2=3.4.4=r43hc72bb7e_0"])).toEqual([
      { name: "r-ggplot2", version: "3.4.4=r43hc72bb7e_0" },
    ]);
  });

  it.each(["", "   ", "=1.17"])("refuses %j, which names no package", (entry) => {
    expect(() => parsePackageSpecs([entry])).toThrow(GalaxyValidationError);
    expect(() => parsePackageSpecs([entry])).toThrow("expected 'name' or 'name=version'");
  });

  // Python's message is `f"invalid package entry {pkg!r}: ..."`, and every expected string here
  // is one the interpreter printed for that f-string.
  it.each([
    ["\n", "invalid package entry '\\n': expected 'name' or 'name=version'"],
    ["=o'k", "invalid package entry \"=o'k\": expected 'name' or 'name=version'"],
    ["\t=1", "invalid package entry '\\t=1': expected 'name' or 'name=version'"],
    ["\u0085", "invalid package entry '\\x85': expected 'name' or 'name=version'"],
    ["", "invalid package entry '': expected 'name' or 'name=version'"],
  ])("reports %j the way repr writes it", (entry, message) => {
    // Interpolating the entry raw put a literal newline in the message and quoted an entry
    // holding an apostrophe with apostrophes -- a different string than the tool returns.
    let thrown: unknown;
    try {
      parsePackageSpecs([entry]);
    } catch (e) {
      thrown = e;
    }
    expect(thrown).toBeInstanceOf(GalaxyValidationError);
    expect((thrown as Error).message).toBe(message);
  });

  it("refuses an empty list", () => {
    expect(() => parsePackageSpecs([])).toThrow("at least one conda package name");
  });
});

describe("recommend_biocontainer", () => {
  it("returns exactly the keys Python's tool returns", async () => {
    stubQuay({ samtools: ["1.17--h00cdaf9_0", "1.16--h0_0"] });
    const out = await recommendBiocontainer({ packages: ["samtools=1.17"] }, ctx);
    expect(Object.keys(out).sort()).toEqual([
      "found",
      "image",
      "match_quality",
      "notes",
      "source",
      "verified",
    ]);
    expect(out).toEqual({
      image: "quay.io/biocontainers/samtools:1.17--h00cdaf9_0",
      found: true,
      match_quality: "exact_version",
      source: "quay_single",
      notes: [],
      verified: true,
    });
  });

  it("resolves several packages to a mulled-v2 image", async () => {
    const repo = v2RepoName(["bwa", "samtools"]);
    const vh = v2VersionHash([
      { name: "bwa", version: "0.7.17" },
      { name: "samtools", version: "1.17" },
    ])!;
    stubQuay({ [repo]: [`${vh}-0`] });
    const out = await recommendBiocontainer({ packages: ["bwa=0.7.17", "samtools=1.17"] }, ctx);
    expect(out.image).toBe(`quay.io/biocontainers/${repo}:${vh}-0`);
    expect(out.source).toBe("quay_mulled_v2");
    expect(out.match_quality).toBe("exact_version");
    expect(out.verified).toBe(true);
  });

  it("says not_found, and leaves verified unchecked, when nothing resolves", async () => {
    stubQuay({});
    const out = await recommendBiocontainer({ packages: ["nopackage"] }, ctx);
    expect(out).toEqual({
      image: null,
      found: false,
      match_quality: "not_found",
      source: "none",
      notes: ["no biocontainer found for 'nopackage'"],
      verified: null,
    });
  });

  it("leaves verified null when the tag list cannot be fetched a second time", async () => {
    // The recommendation is cached, so the verification call is the one that fails: a blip
    // must not turn a resolved image into an unverified-and-therefore-broken one.
    let call = 0;
    vi.stubGlobal("fetch", async () => {
      call += 1;
      if (call === 1) return okText(quayRepo(["1.17--h0_0"]));
      throw new Error("ECONNRESET");
    });
    const out = await recommendBiocontainer({ packages: ["samtools=1.17"] }, ctx);
    expect(out.image).toBe("quay.io/biocontainers/samtools:1.17--h0_0");
    expect(out.verified).toBeNull();
  });

  it("is false for an image whose tag the repository does not have", async () => {
    // A name_only fallback still verifies: the tag it fell back to is a real one.
    stubQuay({ samtools: ["1.17--h0_0"] });
    const out = await recommendBiocontainer({ packages: ["samtools=9.9"] }, ctx);
    expect(out.match_quality).toBe("name_only");
    expect(out.notes).toEqual([
      "requested version 9.9 not found; newest available tag is 1.17--h0_0",
    ]);
    expect(out.verified).toBe(true);
  });

  it("refuses an entry that names no package, before asking quay.io anything", async () => {
    const asked: string[] = [];
    stubQuay({ samtools: ["1.17--h0_0"] }, asked);
    await expect(recommendBiocontainer({ packages: ["=1.17"] }, ctx)).rejects.toBeInstanceOf(
      GalaxyValidationError,
    );
    expect(asked).toEqual([]);
  });

  it("refuses a version that would not survive a shell, before asking quay.io", async () => {
    // Python's build_target builds a CondaTarget, which validates before any request, so
    // "samtools=1.17 2" is a refusal and not a quietly successful newest-tag fallback.
    const asked: string[] = [];
    stubQuay({ samtools: ["1.17--h0_0"] }, asked);
    await expect(
      recommendBiocontainer({ packages: ["samtools=1.17 2"] }, ctx),
    ).rejects.toBeInstanceOf(GalaxyValidationError);
    await expect(recommendBiocontainer({ packages: ["samtools=1.17 2"] }, ctx)).rejects.toThrow(
      "Invalid version [1.17 2] encountered.",
    );
    expect(asked).toEqual([]);
  });

  it("never reports a fabricated tag as found and verified", async () => {
    // A 200 whose tag list is an array, not a map: read leniently, Object.keys turns it into
    // the tag "0" and the tool answers a confident, verified quay.io/biocontainers/samtools:0
    // that was never built. It has to fail instead.
    vi.stubGlobal("fetch", async () =>
      okText('{"namespace":"biocontainers","tags":["1.17--h0_0"]}'),
    );
    const out = await recommendBiocontainer({ packages: ["samtools"] }, ctx).catch(
      (e: unknown) => e as Error,
    );
    expect(out).toBeInstanceOf(Error);
    expect((out as Error).message).toContain("Unexpected response from quay.io");
  });

  it("resolves a version wrapped in U+0085, which Python strips", async () => {
    const asked = stubQuay({ samtools: ["1.17--h0_0", "1.16--h0_0"] });
    const out = await recommendBiocontainer({ packages: ["samtools=\u00851.17\u0085"] }, ctx);
    expect(asked[0]).toContain("/biocontainers/samtools");
    expect(out.image).toBe("quay.io/biocontainers/samtools:1.17--h0_0");
    expect(out.match_quality).toBe("exact_version");
    expect(out.notes).toEqual([]);
  });

  it("reports name_only for a version wrapped in U+FEFF, which Python keeps", async () => {
    // The mirror of the case above: trim would have removed the U+FEFF and turned a
    // name_only answer with a warning into a confident exact_version match.
    stubQuay({ samtools: ["1.17--h0_0"] });
    const out = await recommendBiocontainer({ packages: ["samtools=\ufeff1.17\ufeff"] }, ctx);
    expect(out.match_quality).toBe("name_only");
    expect(out.notes).toEqual([
      "requested version \ufeff1.17\ufeff not found; newest available tag is 1.17--h0_0",
    ]);
  });

  it("refuses an empty list at the schema, in Python's words", () => {
    const parsed = recommendBiocontainerOp.input.packages.safeParse([]);
    expect(parsed.success).toBe(false);
    expect(parsed.success === false && parsed.error.issues[0]?.message).toBe(
      "packages must contain at least one conda package name",
    );
  });

  it("is a read-only tools op named as Python names it", () => {
    expect(recommendBiocontainerOp.name).toBe("recommend_biocontainer");
    expect(recommendBiocontainerOp.domain).toBe("tools");
    expect(recommendBiocontainerOp.readOnly).not.toBe(false);
    expect(recommendBiocontainerOp.requires).toBeUndefined();
  });

  it("projects Python's message either way", () => {
    const found = recommendBiocontainerOp.project!(
      { image: "quay.io/biocontainers/samtools:1.17", match_quality: "exact_version" } as never,
      { packages: ["samtools=1.17"] },
    );
    expect(found.message).toBe("Resolved quay.io/biocontainers/samtools:1.17 (exact_version)");
    const missing = recommendBiocontainerOp.project!(
      { image: null, match_quality: "not_found" } as never,
      { packages: ["nopackage"] },
    );
    expect(missing.message).toBe("No biocontainer found for the requested packages");
  });
});
