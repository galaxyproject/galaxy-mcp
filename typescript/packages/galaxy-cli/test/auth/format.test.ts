// The small formatting helpers behind `auth status`, at their exact boundaries: how long a login
// has left, how much of a token is shown, and how an expiry is put into words. Tables, so each
// edge is one visible row.
import { describe, expect, it } from "vitest";
import { formatTimeLeft, maskToken } from "../../src/auth/commands/common";
import { checkExpiry, describeExpiry, formatExpiry } from "../../src/auth/expiry";

const SEC = 1000;
const MIN = 60 * SEC;
const HOUR = 60 * MIN;
// A duration, unlike the harness's `DAY`, which is a timestamp.
const DAY_MS = 24 * HOUR;

describe("formatTimeLeft", () => {
  // Each row sits on one side of a boundary, and the text is written out by hand and not computed,
  // so a shifted boundary (a minute that rounds up, a day that keeps its minutes) moves one row.
  // Negative and zero both read "under a minute", since the caller never wants a negative time.
  const rows: [what: string, ms: number, text: string][] = [
    ["nothing left", 0, "under a minute"],
    ["59 s", 59 * SEC, "under a minute"],
    ["59.999 s, the last instant before a minute", MIN - 1, "under a minute"],
    ["negative, 1 ms past", -1, "under a minute"],
    ["negative, an hour past", -HOUR, "under a minute"],
    ["60 s", MIN, "1m"],
    ["61 s", MIN + SEC, "1m"],
    ["2 minutes less 1 ms", 2 * MIN - 1, "1m"],
    ["2 minutes", 2 * MIN, "2m"],
    ["59 minutes", 59 * MIN, "59m"],
    ["an hour less 1 ms", HOUR - 1, "59m"],
    ["60 minutes", HOUR, "1h 0m"],
    ["61 minutes", HOUR + MIN, "1h 1m"],
    ["7 h 58 m", 7 * HOUR + 58 * MIN, "7h 58m"],
    ["23 h 59 m", 23 * HOUR + 59 * MIN, "23h 59m"],
    ["a day less 1 ms", DAY_MS - 1, "23h 59m"],
    ["24 h", DAY_MS, "1d 0h"],
    ["25 h 30 m (the minutes are dropped from a day on)", 25 * HOUR + 30 * MIN, "1d 1h"],
    ["48 h", 2 * DAY_MS, "2d 0h"],
    ["10 d 5 h", 10 * DAY_MS + 5 * HOUR, "10d 5h"],
  ];
  it.each(rows)("%s", (_what, ms, text) => {
    expect(formatTimeLeft(ms)).toBe(text);
  });
});

describe("maskToken", () => {
  const LETTERS = "abcdefghijklmnopqrstuvwxyz0123456789";
  const STARS = "*".repeat(8);
  // Up to 12 characters, none of the token shows. From 13, the last four do, so a person can tell
  // tokens apart without a short token giving away a third of itself.
  const rows: [length: number, shown: string][] = [
    [0, STARS],
    [1, STARS],
    [4, STARS],
    [8, STARS],
    [12, STARS],
    [13, `${STARS}jklm`],
    [14, `${STARS}klmn`],
    [36, `${STARS}6789`],
  ];
  it.each(rows)("a token of %i characters shows %s", (length, shown) => {
    const token = LETTERS.slice(0, length);
    expect(token).toHaveLength(length);
    expect(maskToken(token)).toBe(shown);
  });
  it("never shows more than the last four characters", () => {
    const token = "SECRET-PREFIX-0123456789-tail";
    const masked = maskToken(token);
    expect(masked).toBe(`${STARS}tail`);
    expect(masked).not.toContain("SECRET");
  });
});

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
