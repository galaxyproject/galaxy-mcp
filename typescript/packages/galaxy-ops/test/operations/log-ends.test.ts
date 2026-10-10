import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { logEnds } from "../../src/operations/log-ends";

const line = (i: number) => `line ${String(i).padStart(3, "0")} of a long log`;
const lines = (from: number, to: number) => Array.from({ length: to - from }, (_, k) => line(from + k)).join("\n");

/**
 * The same table as python/tests/test_job_operations.py LOG_ENDS_VECTORS, as literal expected
 * strings: the two clamps are held to the same bytes, and a loose assertion here would let
 * them drift apart where the goldens do not look.
 */
const VECTORS: Array<[name: string, text: string, budget: number, expected: string]> = [
  ["short, untouched", "a\nb\n", 4096, "a\nb\n"],
  ["empty at any budget", "", 5, ""],
  ["budget 0 is uncut", "x".repeat(100), 0, "x".repeat(100)],
  [
    "ASCII lines at the default budget",
    lines(0, 300),
    4096,
    `${lines(0, 89)}\n[... 2807 of 6899 bytes omitted ...]\n${lines(211, 300)}`,
  ],
  [
    "3-byte text with no newline: both cuts land mid-character",
    "€".repeat(40),
    64,
    `${"€".repeat(10)}\n[... 60 of 120 bytes omitted ...]\n${"€".repeat(10)}`,
  ],
  [
    "newline in each half, through multi-byte text",
    Array.from({ length: 12 }, () => "€".repeat(7)).join("\n"),
    64,
    `${"€".repeat(7)}\n[... 221 of 263 bytes omitted ...]\n${"€".repeat(7)}`,
  ],
  [
    "4-byte text with no newline",
    "🎉".repeat(50),
    64,
    `${"🎉".repeat(8)}\n[... 136 of 200 bytes omitted ...]\n${"🎉".repeat(8)}`,
  ],
  [
    "mixed widths, and a tail that is empty because the newline is back's last byte",
    "a€b🎉c\n".repeat(30),
    11,
    "a€b\n[... 325 of 330 bytes omitted ...]\n",
  ],
  [
    "a BOM at the start of the tail is kept",
    `${"x".repeat(60)}﻿${"y".repeat(40)}`,
    86,
    `${"x".repeat(43)}\n[... 17 of 103 bytes omitted ...]\n﻿${"y".repeat(40)}`,
  ],
  [
    "CRLF: the cut is on the newline, so the head ends in a bare CR",
    Array.from({ length: 40 }, (_, i) => `l${i}`).join("\r\n"),
    32,
    "l0\r\nl1\r\nl2\r\nl3\r\n[... 160 of 188 bytes omitted ...]\nl37\r\nl38\r\nl39",
  ],
  ["budget 1: halves of 0, both ends empty", "abc\n", 1, "\n[... 4 of 4 bytes omitted ...]\n"],
  [
    "odd budget: one byte unused on each side",
    "a".repeat(100),
    33,
    `${"a".repeat(16)}\n[... 68 of 100 bytes omitted ...]\n${"a".repeat(16)}`,
  ],
  [
    "a newline only at byte 0: the head is empty",
    `\n${"b".repeat(100)}`,
    20,
    `\n[... 91 of 101 bytes omitted ...]\n${"b".repeat(10)}`,
  ],
];

describe("logEnds", () => {
  it.each(VECTORS)("%s", (_name, text, budget, expected) => {
    const got = logEnds(text, budget);
    expect(got).toBe(expected);
    expect(got).not.toContain("�");
  });

  it("hands an untouched log back as the same string", () => {
    const text = "short";
    expect(logEnds(text, 0)).toBe(text);
    expect(logEnds(text, 4096)).toBe(text);
  });

  it("counts a lone surrogate as the three bytes TextEncoder writes, as Python does", () => {
    // JSON.parse turns a "\ud800" escape into a lone surrogate; TextEncoder writes EF BF BD
    // for it, and the Python side substitutes U+FFFD before encoding so the counts agree.
    const text = "\ud800".repeat(30);
    expect(logEnds(text, 12)).toBe(`${"�".repeat(2)}\n[... 78 of 90 bytes omitted ...]\n${"�".repeat(2)}`);
  });

  it("produces the Python golden's bytes from the Python golden's input", () => {
    // The multibyte case is the proof that the two clamps agree where PR-era code did not:
    // the reply table is fed through this clamp and compared with what server.py wrote.
    const dir = join(__dirname, "..", "..", "..", "..", "..", "contract", "envelopes", "get_job_logs");
    const replies = JSON.parse(readFileSync(join(dir, "multibyte_text_at_the_cut.galaxy.json"), "utf8"));
    const golden = JSON.parse(readFileSync(join(dir, "multibyte_text_at_the_cut.json"), "utf8"));
    const reply = (replies.routes ?? replies)[0].body as Record<string, string>;
    const budget = 64;
    for (const field of Object.keys(golden.data)) {
      const got = logEnds(reply[field]!, budget);
      expect(got, field).toBe(golden.data[field]);
      expect(Buffer.from(got, "utf8").equals(Buffer.from(golden.data[field], "utf8")), field).toBe(true);
      expect(got).not.toContain("�");
    }
  });
});
