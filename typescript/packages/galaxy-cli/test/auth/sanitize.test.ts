import { describe, expect, it } from "vitest";
import { loginCommand } from "../../src/auth/commands/login";
import { logoutCommand } from "../../src/auth/commands/logout";
import { statusCommand } from "../../src/auth/commands/status";
import { sanitize } from "../../src/auth/sanitize";
import { AuthError } from "../../src/auth/types";
import { fakeLogin, harness } from "./harness";

const HOST_ARG = "https://dev.galaxy.example.org";
// Our own multi-line output keeps its newlines; everything else a terminal could act on must go.
// The ranges are written out here, apart from `sanitize`'s own patterns, so the check does not
// share a mistake with the code it checks.
const CONTROL_EXCEPT_NEWLINE = /[\u0000-\u0009\u000b-\u001f\u007f-\u009f]/;

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

// Each case drives a command that prints a different piece of text the server controls, through the
// harness's fakes, and reads what it wrote. The text is an escape sequence that retitles the window
// and clears the screen. `STRIPPED` is what is left once the control characters are removed, worked
// out by hand: the bytes after an ESC stay as harmless text. A new place that prints server text
// without sanitising it would need its own case here.
describe("server text is sanitised before a command prints it", () => {
  const EVIL = "\u001b]0;pwned\u0007\u001b[2J";
  const STRIPPED = "]0;pwned[2J";

  it("login: the user code, the page and the username", async () => {
    const h = harness();
    h.server.pending.userCode = `AB${EVIL}CD`;
    h.server.pending.verificationUri = `https://dev.galaxy.example.org/${EVIL}device`;
    h.knobs.whoami = { username: `evil${EVIL}user`, email: "e@example.org" };
    expect(await loginCommand({ host: HOST_ARG }, h.deps)).toBe(0);
    expect(h.cap.err).not.toMatch(CONTROL_EXCEPT_NEWLINE);
    expect(h.cap.err).toContain(`AB${STRIPPED}CD`);
    expect(h.cap.err).toContain(`as evil${STRIPPED}user`);
  });

  // 76 is sysexits' EX_PROTOCOL, the code for a `protocol` error: the server's answer was one the
  // flow cannot use. The error's own text quotes the server, and it must come out clean too.
  it("login: an error that quotes the server", async () => {
    const h = harness();
    h.knobs.discoverError = new AuthError("protocol", `refused: ${EVIL}`);
    expect(await loginCommand({ host: HOST_ARG }, h.deps)).toBe(76);
    expect(h.cap.err).not.toMatch(CONTROL_EXCEPT_NEWLINE);
  });

  it("status: the stored username and the text of a failed check", async () => {
    const h = harness({ stored: [fakeLogin({ username: `st${EVIL}ored` })] });
    h.knobs.whoami = new AuthError("not_logged_in", `bad${EVIL}key`);
    await statusCommand({}, h.deps);
    expect(h.cap.err).not.toMatch(CONTROL_EXCEPT_NEWLINE);
  });

  it("status --json: the username Galaxy reports", async () => {
    const h = harness({ stored: [fakeLogin()] });
    h.knobs.whoami = { username: `u${EVIL}x` };
    await statusCommand({ json: true }, h.deps);
    expect((JSON.parse(h.cap.out) as { username?: string }[])[0]?.username).toBe(`u${STRIPPED}x`);
  });

  it("logout: the text of a failed revocation", async () => {
    const h = harness({ stored: [fakeLogin()] });
    h.server.revokeError = new Error(`no${EVIL}pe`);
    await logoutCommand({}, h.deps);
    expect(h.cap.err).not.toMatch(CONTROL_EXCEPT_NEWLINE);
    expect(h.cap.err).toContain(`no${STRIPPED}pe`);
  });
});
