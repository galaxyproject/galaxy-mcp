// The expiry helpers at their exact boundaries: when a login counts as expired and how an expiry is
// put into words. Tables, so each edge is one visible row.
import { describe, expect, it } from "vitest";
import { checkExpiry, describeExpiry, formatExpiry } from "../../src/auth/expiry";

const SEC = 1000;
const MIN = 60 * SEC;

describe("describeExpiry and formatExpiry at the edges", () => {
  const now = new Date("2026-11-06T12:00:00.000Z");
  const at = (ms: number) => new Date(now.getTime() + ms).toISOString();

  // The clause says "about to expire" for a login inside the 60 s margin but still in the future,
  // and "expired at" from the instant of expiresAt on, including exactly now.
  const rows: [what: string, offset: number, start: string][] = [
    ["1 ms past", -1, "expired at "],
    ["exactly now", 0, "expired at "],
    ["1 ms ahead", 1, "is about to expire (at "],
    ["59 s ahead", 59 * SEC, "is about to expire (at "],
    ["60 s ahead, the margin", 60 * SEC, "is about to expire (at "],
  ];
  it.each(rows)("expiresAt %s: %s", (_what, offset, start) => {
    expect(describeExpiry(at(offset), now).startsWith(start)).toBe(true);
  });

  it("names the time in UTC to the minute", () => {
    expect(describeExpiry(at(-5 * MIN), now)).toBe("expired at 2026-11-06 11:55 UTC");
    expect(describeExpiry(at(30 * SEC), now)).toBe("is about to expire (at 2026-11-06 12:00 UTC)");
  });

  it("says nothing is set, not unreadable, for null and undefined", () => {
    expect(checkExpiry(null, now)).toEqual({ state: "none" });
    expect(checkExpiry(undefined, now)).toEqual({ state: "none" });
  });

  // A hand-edited hosts.yml must not flood the terminal: 40 characters of the bad value, no more.
  // The first 40 characters are `A` and any beyond are `B`, so where the value is cut shows in the
  // quoted text without computing it from the code under test.
  const unreadable = (n: number) => "A".repeat(Math.min(n, 40)) + "B".repeat(Math.max(0, n - 40));
  const lengths: [length: number, quoted: string][] = [
    [1, "A"],
    [39, "A".repeat(39)],
    [40, "A".repeat(40)],
    [41, "A".repeat(40)],
    [100, "A".repeat(40)],
  ];
  it.each(lengths)("an unreadable value of %i characters is quoted as %s", (n, quoted) => {
    const value = unreadable(n);
    expect(describeExpiry(value, now)).toBe(
      `has an unreadable expiry time (${quoted}), so it cannot be trusted`,
    );
    expect(formatExpiry(value)).toBe(`an unreadable time (${quoted})`);
  });
});
