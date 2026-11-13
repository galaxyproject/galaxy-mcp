import { describe, expect, it } from "vitest";
import { sanitize } from "../../src/auth/sanitize";

describe("sanitize", () => {
  it("removes ESC, C0, DEL and C1 controls, including CR, LF and tab", () => {
    expect(sanitize("a\u001b]0;x\u0007b\u001b[2Jc\r\nd\u007fe\u009bf\tg")).toBe("a]0;xb[2Jcdefg");
  });

  // A sanitiser that stripped everything outside ASCII would mangle real names, which Galaxy
  // usernames and emails can contain.
  it("keeps ordinary text, symbols and accented or non-Latin letters", () => {
    expect(sanitize("ordinary text, ✓ and é")).toBe("ordinary text, ✓ and é");
    expect(sanitize("José Müller Łukasz 李")).toBe("José Müller Łukasz 李");
  });

  it("can keep newlines for our own multi-line messages", () => {
    expect(sanitize("a\nb\u001b[0m\r", { keepNewlines: true })).toBe("a\nb[0m");
  });

  it.each([
    "\u061c",
    "\u200b",
    "\u200e",
    "\u200f",
    "\u202a",
    "\u202e",
    "\u2060",
    "\u2064",
    "\u2066",
    "\u2069",
    "\ufeff",
    "\u00ad",
    "\u200d",
  ])("removes the invisible format character %j, which can reorder or hide text", (c) => {
    expect(sanitize(`a${c}b`)).toBe("ab");
    expect(sanitize(`a${c}b`, { keepNewlines: true })).toBe("ab");
  });
});
