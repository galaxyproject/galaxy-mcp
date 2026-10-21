import { describe, expect, it } from "vitest";
import { historiesAppHtml } from "../src/apps/histories";

describe("the histories card", () => {
  const script = historiesAppHtml.match(/<script>([\s\S]*)<\/script>/)?.[1] ?? "";

  it("is one self-contained page", () => {
    expect(historiesAppHtml).toMatch(/^<!doctype html>/);
    // Nothing loaded from elsewhere: the host's sandbox would block it, and the card needs none.
    expect(historiesAppHtml).not.toMatch(/\b(src|href)=|@import|url\(/);
  });

  it("carries a script that parses", () => {
    expect(script).toContain('request("ui/initialize"');
    expect(() => new Function(script)).not.toThrow();
  });

  it("reads Galaxy's zone-less times as UTC", () => {
    const pattern = /[zZ]|[+-]\d\d:?\d\d$/;
    expect(script).toContain(String.raw`/[zZ]|[+-]\d\d:?\d\d$/`);
    expect(pattern.test("2026-10-21T12:00:00.123456")).toBe(false);
    expect(pattern.test("2026-10-21T12:00:00Z")).toBe(true);
    expect(pattern.test("2026-10-21T12:00:00+02:00")).toBe(true);
  });
});
