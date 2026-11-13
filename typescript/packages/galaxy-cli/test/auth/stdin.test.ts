import { PassThrough } from "node:stream";
import { describe, expect, it } from "vitest";
import {
  InputCancelled,
  MAX_KEY_BYTES,
  readPiped,
  readSecret,
  type RawStdin,
} from "../../src/auth/stdin";
import { AuthError } from "../../src/auth/types";

// Input that cannot be an API key, which is printable ASCII with no whitespace: a space, a curly
// quote and a non-ASCII letter, the marks of a broken paste. Both readers refuse them as a usage
// error, and the message must not repeat the input, or a pasted secret would reach the screen and
// any log of the terminal.
const NOT_KEYS = [
  ["a space", "fake key"],
  ["a curly quote", "fake‘key"],
  ["a non-ASCII letter", "fakékey"],
];

describe("readPiped", () => {
  /** A signal that never aborts, for the tests that are not about cancelling. */
  const signal = () => new AbortController().signal;
  /** A stream already given `text` and ended, as if the caller's `echo key |` had finished. */
  const piped = (text: string) => {
    const s = new PassThrough();
    s.end(text);
    return s;
  };

  it("returns what was piped", async () => {
    await expect(readPiped(piped("fake-key-1\n"), signal())).resolves.toBe("fake-key-1\n");
  });

  it.each(NOT_KEYS)(
    "refuses a key with %s as a usage error, without echoing it",
    async (_what, bad) => {
      const e = await readPiped(piped(`${bad}\n`), signal()).catch((x: unknown) => x);
      expect(e).toBeInstanceOf(AuthError);
      expect((e as AuthError).kind).toBe("usage");
      expect((e as AuthError).message).not.toContain(bad);
    },
  );

  // `k` is one byte, so the length is the byte count. Nothing near the cap can be a key (see
  // `MAX_KEY_BYTES`), so a file piped in by mistake is refused with a message that says so. The
  // pair of cases pins the boundary itself.
  it("accepts input right at the size cap and refuses one byte more", async () => {
    await expect(readPiped(piped("k".repeat(MAX_KEY_BYTES)), signal())).resolves.toHaveLength(
      MAX_KEY_BYTES,
    );
    const e = await readPiped(piped("k".repeat(MAX_KEY_BYTES + 1)), signal()).catch(
      (x: unknown) => x,
    );
    expect((e as AuthError).kind).toBe("usage");
  });

  // The stream is never ended, so only the abort can finish the read. A `data` listener left on
  // stdin would keep it flowing, and the process alive, after a login that was cancelled.
  it("stops reading, leaving no listener behind, when the signal aborts", async () => {
    const s = new PassThrough();
    const ac = new AbortController();
    const p = readPiped(s, ac.signal);
    ac.abort(new Error("ctrl-c"));
    await expect(p).rejects.toThrow("ctrl-c");
    expect(s.listenerCount("data")).toBe(0);
  });
});

describe("readSecret", () => {
  /**
   * A fake terminal: a stream the test writes keystrokes to, with a `setRawMode` that records each
   * value it is called with in `raw`. A real terminal in raw mode has no echo and no line editing, so
   * a test can tell from `raw` that the terminal went raw (`true` first) and was put back (`false`
   * last) on every way out.
   */
  function terminal() {
    const raw: boolean[] = [];
    const s = Object.assign(new PassThrough(), { setRawMode: (m: boolean) => void raw.push(m) });
    return { s, raw, stdin: s as unknown as RawStdin };
  }
  /** A signal that never aborts, for the tests that are not about cancelling. */
  const signal = () => new AbortController().signal;

  it("reads a line without echo and puts the terminal back", async () => {
    const { s, raw, stdin } = terminal();
    const p = readSecret(stdin, signal());
    s.write("fake-key\r");
    await expect(p).resolves.toBe("fake-key");
    expect(raw[0]).toBe(true);
    expect(raw.at(-1)).toBe(false);
  });

  it("finishes on Ctrl-D too", async () => {
    const { s, stdin } = terminal();
    const p = readSecret(stdin, signal());
    s.write("fake-key\x04");
    await expect(p).resolves.toBe("fake-key");
  });

  // The two backspaces (0x7f) remove `XX`, so the expected `abc` is worked out by hand from the keystrokes.
  it("honours backspace", async () => {
    const { s, stdin } = terminal();
    const p = readSecret(stdin, signal());
    s.write("abXX\x7f\x7fc\n");
    await expect(p).resolves.toBe("abc");
  });

  // Arrow, Home and Delete keys are escape sequences, and in application mode `ESC O x`. Left in,
  // the letters of them would become part of the key: `ESC O A` is the three characters `OA`.
  it.each([
    ["CSI arrow keys", "ab\x1b[A\x1b[Dc\x1b[3~"],
    ["application-mode arrow keys", "ab\x1bOA\x1bODc"],
    ["bracketed paste markers", "\x1b[200~abc\x1b[201~"],
  ])("ignores %s", async (_what, keys) => {
    const { s, stdin } = terminal();
    const p = readSecret(stdin, signal());
    s.write(`${keys}\n`);
    await expect(p).resolves.toBe("abc");
  });

  it("backspaces one character, not half of one", async () => {
    const { s, stdin } = terminal();
    const p = readSecret(stdin, signal());
    s.write("a\u{1F600}\x7fb\n");
    await expect(p).resolves.toBe("ab");
  });

  it("returns nothing, for the caller to explain, on Enter or Ctrl-D with nothing typed", async () => {
    for (const end of ["\n", "\x04"]) {
      const { s, stdin } = terminal();
      const p = readSecret(stdin, signal());
      s.write(end);
      await expect(p).resolves.toBe("");
    }
  });

  // The cap is on bytes, as `MAX_KEY_BYTES` says: this many four-byte characters are 4 times over it.
  it("refuses more than the limit in bytes, not in characters", async () => {
    const { s, raw, stdin } = terminal();
    const p = readSecret(stdin, signal()).catch((x: unknown) => x);
    s.write("\u{1F600}".repeat(MAX_KEY_BYTES / 4 + 1));
    const e = await p;
    expect(e).toBeInstanceOf(AuthError);
    expect((e as AuthError).message).toContain("16 KiB");
    expect(raw.at(-1)).toBe(false);
  });

  it("is cancelled by Ctrl-C, which raw mode delivers as a character", async () => {
    const { s, raw, stdin } = terminal();
    const p = readSecret(stdin, signal());
    s.write("fake\x03");
    await expect(p).rejects.toBeInstanceOf(InputCancelled);
    expect(raw.at(-1)).toBe(false);
  });

  it("puts the terminal back when the signal aborts", async () => {
    const { raw, stdin } = terminal();
    const ac = new AbortController();
    const p = readSecret(stdin, ac.signal);
    ac.abort(new Error("ctrl-c"));
    await expect(p).rejects.toThrow("ctrl-c");
    expect(raw.at(-1)).toBe(false);
  });

  it.each(NOT_KEYS)("refuses a key with %s, without echoing it", async (_what, bad) => {
    const { s, stdin } = terminal();
    const p = readSecret(stdin, signal()).catch((x: unknown) => x);
    s.write(`${bad}\r`);
    const e = await p;
    expect(e).toBeInstanceOf(AuthError);
    expect((e as AuthError).message).not.toContain(bad);
  });
});
