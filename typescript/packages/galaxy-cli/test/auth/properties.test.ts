// Property tests (fast-check) where a canonical form or a round trip is the contract.
// Fixed seed and bounded runs keep them deterministic and quick; a failure prints the
// counterexample and the seed, so it can be replayed and then pinned as an example.
import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { normalizeBaseUrl } from "../../src/auth/base-url";
import { sanitize } from "../../src/auth/sanitize";
import { AuthError } from "../../src/auth/types";

// The seed is fixed, so a failure on one machine is a failure on every machine and can be replayed
// from the seed in the report. Two hundred runs is enough to cover the generators below in well
// under a second.
const params = { seed: 20260707, numRuns: 200 };

// Small generators: a mix of well-formed URLs and the awkward pieces that matter to the key.
const host = fc.constantFrom(
  "host.example",
  "HOST.Example:443",
  "h.example:8443",
  "localhost:8080",
  "127.0.0.1",
  "[::1]:9",
  "evil.example",
);
const segment = fc.constantFrom("api", "galaxy", "apiary", "v2", "", "%61pi");
const pathPart = fc
  .array(segment, { maxLength: 4 })
  .map((parts) => parts.map((p) => `/${p}`).join(""));
const tail = fc.constantFrom("", "/", "//", "?x=1", "#f", "/?q#f");
const scheme = fc.constantFrom(
  "",
  "https://",
  "http://",
  "HTTPS://",
  "ftp://",
  "user@",
  "https://u:p@",
);
const urlish = fc.oneof(
  fc.tuple(scheme, host, pathPart, tail).map(([s, h, p, t]) => `${s}${h}${p}${t}`),
  fc.string({ maxLength: 30 }),
);

/** The key for `input`, or undefined when it was refused as a bad URL. Any other error propagates. */
const attempt = (input: string) => {
  try {
    return normalizeBaseUrl(input);
  } catch (e) {
    if (!(e instanceof AuthError)) throw e; // anything but an AuthError escaping fails the property
    return undefined;
  }
};

describe("normalizeBaseUrl properties", () => {
  it("throws only AuthError, and the key has no userinfo, query, fragment or trailing slash, and is http only for loopback", () => {
    fc.assert(
      fc.property(urlish, (input) => {
        const key = attempt(input);
        if (key === undefined) return;
        const url = new URL(key);
        expect(url.username + url.password + url.search + url.hash).toBe("");
        expect(key.endsWith("/")).toBe(false);
        expect(key).not.toMatch(/[?#]/);
        expect(["http:", "https:"]).toContain(url.protocol);
        // Only http needs a loopback host; for https any host will do.
        expect(url.protocol === "http:" ? url.hostname : "localhost").toMatch(
          /^(?:localhost|\[::1\]|127(?:\.\d{1,3}){3})$/,
        );
      }),
      params,
    );
  });

  // The key is a credential key, and a stored key is normalised again on lookup, so it must come
  // back unchanged. A doubled `/api` broke this: it normalised to `/api` and then, on the second
  // pass, to the bare host. A longer run of this property found that, and it is now pinned with
  // examples in units.test.ts.
  it("is idempotent: a stored key normalises to itself", () => {
    fc.assert(
      fc.property(urlish, (input) => {
        const key = attempt(input);
        expect(key === undefined ? undefined : normalizeBaseUrl(key)).toBe(key);
      }),
      params,
    );
  });
});

describe("sanitize", () => {
  // The expected property uses the Unicode general categories (`\p{Cc}`, `\p{Cf}`), which are not
  // how `sanitize` is written: it lists ranges by hand, so a character it forgot would show as a
  // counterexample. Code points up to U+20FF cover C0, C1 and the common Cf ranges; the extras are
  // Cf characters beyond them.
  const unit = fc.oneof(
    fc.integer({ min: 0, max: 0x20ff }).map((c) => String.fromCodePoint(c)),
    fc.constantFrom("\ufeff", "\u{e0041}", "\u{1d173}"),
  );

  it("leaves no control or format characters, and is idempotent", () => {
    fc.assert(
      fc.property(fc.string({ unit, maxLength: 40 }), fc.boolean(), (value, keepNewlines) => {
        const once = sanitize(value, { keepNewlines });
        expect(keepNewlines ? once.replace(/\n/g, "") : once).not.toMatch(/[\p{Cc}\p{Cf}]/u);
        expect(sanitize(once, { keepNewlines })).toBe(once);
      }),
      params,
    );
  });
});
