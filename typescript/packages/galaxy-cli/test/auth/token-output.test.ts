/**
 * `auth token`, whose job is to print the token. On it, stdout carries the secret and stderr
 * carries every message about it.
 */
import { describe, it, expect } from "vitest";
import { tokenCommand } from "../../src/auth/commands/token";
import { DAY, FAKE_TOKEN, HOST, OTHER, SOON, fakeLogin, harness } from "./harness";

type H = ReturnType<typeof harness>;
const commands: [string, (h: H) => Promise<number>][] = [
  ["token", (h) => tokenCommand({}, h.deps)],
  ["token --json", (h) => tokenCommand({ json: true }, h.deps)],
];

describe("auth token", () => {
  it("prints only the token and a newline", async () => {
    const h = harness({ stored: [fakeLogin()] });
    expect(await tokenCommand({}, h.deps)).toBe(0);
    expect(h.cap.out).toBe(`${FAKE_TOKEN}\n`);
    expect(h.cap.err).toBe("");
  });

  it("--json prints the versioned object", async () => {
    const h = harness({ stored: [fakeLogin({ expiresAt: SOON })] });
    await tokenCommand({ json: true }, h.deps);
    expect(h.cap.out.endsWith("\n")).toBe(true);
    expect(JSON.parse(h.cap.out)).toEqual({
      version: 1,
      base_url: HOST,
      token: FAKE_TOKEN,
      token_type: "x-api-key",
      expires_at: SOON,
    });
  });

  it("--json has expires_at null for a pasted key", async () => {
    const h = harness({ stored: [fakeLogin()] });
    await tokenCommand({ json: true }, h.deps);
    expect((JSON.parse(h.cap.out) as { expires_at?: unknown }).expires_at).toBeNull();
  });

  it("makes no live call", async () => {
    const h = harness({ stored: [fakeLogin()] });
    await tokenCommand({}, h.deps);
    expect(h.whoamiCalls).toEqual([]);
  });

  it("picks the named host", async () => {
    const h = harness({
      stored: [fakeLogin(), fakeLogin({ baseUrl: OTHER, token: "other-fake-token" })],
    });
    await tokenCommand({ host: OTHER }, h.deps);
    expect(h.cap.out).toBe("other-fake-token\n");
  });
});

// Whoever runs this substitutes stdout (`$(gxy auth token)`), so when there is no token it must stay
// empty and the reason go to stderr, with an exit code. Under a minute left counts as expired, the margin operations use too
// (config.ts), since such a token would run out before the caller used it.
describe.each(commands)("%s when there is nothing to hand out", (_name, run) => {
  it("exits 77 on stderr, with nothing on stdout, when not logged in", async () => {
    const h = harness();
    // a host is known (env) but nothing is stored for it
    h.deps.resolveHost = async () => HOST;
    expect(await run(h)).toBe(77);
    expect(h.cap.out).toBe("");
    expect(h.cap.err).toContain(`gxy auth login --host ${HOST}`);
  });

  it("exits 77 with how to log in when nothing is stored and nothing names a host", async () => {
    const h = harness();
    expect(await run(h)).toBe(77);
    expect(h.cap.out).toBe("");
    expect(h.cap.err).toContain("gxy auth login --host");
  });

  it("exits 77, as about to expire, for a login with under a minute left", async () => {
    const h = harness({
      stored: [fakeLogin({ expiresAt: new Date(new Date(DAY).getTime() + 30_000).toISOString() })],
    });
    expect(await run(h)).toBe(77);
    expect(h.cap.out).toBe("");
    expect(h.cap.err).toMatch(/about to expire/);
  });

  it("exits 77 and says the expiry is unreadable, rather than treating it as none", async () => {
    const h = harness({ stored: [fakeLogin({ expiresAt: "tomorrow" })] });
    expect(await run(h)).toBe(77);
    expect(h.cap.out).toBe("");
    expect(h.cap.err).toMatch(/unreadable expiry time \(tomorrow\)/);
  });

  it("exits 77 and names the expiry when the login has expired", async () => {
    const h = harness({ stored: [fakeLogin({ expiresAt: "2026-11-06T09:00:00.000Z" })] });
    expect(await run(h)).toBe(77);
    expect(h.cap.out).toBe("");
    expect(h.cap.err).toContain("expired at 2026-11-06 09:00 UTC");
    expect(h.cap.err).toContain(`gxy auth login --host ${HOST}`);
    expect(h.cap.err).not.toContain(FAKE_TOKEN);
  });
});

// 64 is EX_USAGE: the caller did not say which Galaxy, and can. 76 is EX_PROTOCOL: an insecure
// `--host` is refused for what it is, not reported as "not logged in", which is 77 (EX_NOPERM).
describe("choosing the host", () => {
  it("exits 64, not 77, when several logins are stored and none is named", async () => {
    const h = harness({ stored: [fakeLogin(), fakeLogin({ baseUrl: OTHER })] });
    expect(await tokenCommand({}, h.deps)).toBe(64);
    expect(h.cap.out).toBe("");
  });

  it("exits 76, not 77, for an insecure --host", async () => {
    const h = harness();
    expect(await tokenCommand({ host: "http://remote.example" }, h.deps)).toBe(76);
  });
});
