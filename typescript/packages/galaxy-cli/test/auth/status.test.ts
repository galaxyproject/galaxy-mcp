/**
 * `status` in process: every stored login is read back and checked live against its Galaxy, all
 * hosts at once, and printed to stderr in stored order. stdout is only for what a script asked for (`--json`,
 * `--show-token`).
 */
import { describe, it, expect, vi } from "vitest";
import { statusCommand } from "../../src/auth/commands/status";
import { createWhoami } from "../../src/auth/whoami";
import { AuthError } from "../../src/auth/types";
import { FAKE_TOKEN, HOST, OTHER, fakeLogin, harness, makeTokenUnreadable } from "./harness";

const expiredLogin = () => fakeLogin({ expiresAt: "2026-11-06T09:00:00.000Z" });
const twoLogins = () =>
  harness({ stored: [fakeLogin(), fakeLogin({ baseUrl: OTHER, username: "bob" })] });

describe("auth status", () => {
  it("shows a live check, storage, method, scopes, time left and a masked token, on stderr", async () => {
    const h = harness({ stored: [fakeLogin()] });
    expect(await statusCommand({}, h.deps)).toBe(0);
    expect(h.cap.out).toBe("");
    expect(h.cap.err).toContain("dev.galaxy.example.org");
    expect(h.cap.err).toContain("Logged in as aioue");
    expect(h.cap.err).toMatch(/Storage\s+system keyring/);
    expect(h.cap.err).toMatch(/Method\s+device login/);
    expect(h.cap.err).toMatch(/Scopes\s+histories:read tools:run/);
    expect(h.cap.err).toMatch(/Expires\s+in 7h/);
    // The token is masked: only its last four characters show.
    expect(h.cap.err).not.toContain(FAKE_TOKEN);
    expect(h.cap.err).toContain(FAKE_TOKEN.slice(-4));
    expect(h.whoamiCalls).toHaveLength(1);
  });

  // `--show-token` is the one way `status` prints a token, and it goes to stdout, where a script
  // reads data, like `auth token`. stderr always carries the masked form, so it stays safe to log.
  it("puts the unmasked token on stdout, never stderr, with --show-token", async () => {
    const h = harness({ stored: [fakeLogin()] });
    await statusCommand({ showToken: true }, h.deps);
    expect(h.cap.out).toBe(`${HOST}\t${FAKE_TOKEN}\n`);
    expect(h.cap.err).not.toContain(FAKE_TOKEN);
  });

  it("says 'no expiry' and 'pasted' for a pasted key", async () => {
    const h = harness({ stored: [fakeLogin({ method: "pasted", scopes: [], expiresAt: null })] });
    await statusCommand({}, h.deps);
    expect(h.cap.err).toMatch(/Expires\s+no expiry/);
    expect(h.cap.err).toMatch(/Method\s+pasted API key/);
  });

  it("lists every stored login, or just the one with --host", async () => {
    const all = twoLogins();
    await statusCommand({}, all.deps);
    expect(all.cap.err).toContain("dev.galaxy.example.org");
    expect(all.cap.err).toContain("other.example.org");

    const one = twoLogins();
    await statusCommand({ host: OTHER }, one.deps);
    expect(one.cap.err).not.toContain("dev.galaxy.example.org");
    expect(one.cap.err).toContain("other.example.org");
  });

  // An expiry the clock already shows needs no request to find out, so none is made: `whoamiCalls`
  // is empty, and the token goes nowhere.
  it("exits 1 and skips the live check for an expired login", async () => {
    const h = harness({ stored: [expiredLogin()] });
    expect(await statusCommand({}, h.deps)).toBe(1);
    expect(h.cap.err).toContain("expired at 2026-11-06 09:00 UTC");
    expect(h.whoamiCalls).toEqual([]);
  });

  it.each([
    ["Galaxy rejects the token", new AuthError("not_logged_in", "Galaxy rejected the token.")],
    [
      "Galaxy cannot be reached",
      new AuthError("network", "Could not reach dev.galaxy.example.org."),
    ],
  ])("exits 1 and says why when %s", async (_when, failure) => {
    const h = harness({ stored: [fakeLogin()] });
    h.knobs.whoami = failure;
    expect(await statusCommand({}, h.deps)).toBe(1);
    expect(h.cap.err).toContain(failure.message);
  });

  it("names the certificate setting when Galaxy cannot be reached securely", async () => {
    const h = harness({ stored: [fakeLogin()] });
    h.deps.whoami = createWhoami(async () => {
      throw new TypeError("fetch failed", {
        cause: Object.assign(new Error("self-signed certificate"), {
          code: "DEPTH_ZERO_SELF_SIGNED_CERT",
        }),
      });
    });
    expect(await statusCommand({}, h.deps)).toBe(1);
    expect(h.cap.err).toContain("NODE_EXTRA_CA_CERTS");
  });

  // What an operation would use for each Galaxy, from `deps.keySourceFor` (config.ts's order, which
  // config.test.ts covers): said at the top when something outranks the logins, per login, and in --json.
  describe("the key an operation would use", () => {
    it("names GALAXY_API_KEY at the top and per login, and puts key_source in --json", async () => {
      const h = harness({ stored: [fakeLogin()], env: { GALAXY_API_KEY: "x" } });
      await statusCommand({}, h.deps);
      expect(h.cap.err).toContain("GALAXY_API_KEY takes precedence over stored logins");
      expect(h.cap.err.indexOf("takes precedence")).toBeLessThan(
        h.cap.err.indexOf(HOST.replace("https://", "")),
      );
      expect(h.cap.err).toContain("Key used by operations: GALAXY_API_KEY");
      const j = harness({ stored: [fakeLogin()], env: { GALAXY_API_KEY: "x" } });
      await statusCommand({ json: true }, j.deps);
      expect((JSON.parse(j.cap.out) as { key_source: string }[])[0]!.key_source).toBe("env");
    });
    it("says stored login, with no warning, when nothing outranks it", async () => {
      const h = harness({ stored: [fakeLogin()] });
      await statusCommand({}, h.deps);
      expect(h.cap.err).not.toContain("takes precedence");
      expect(h.cap.err).toContain("Key used by operations: stored login");
      const j = harness({ stored: [fakeLogin()] });
      await statusCommand({ json: true }, j.deps);
      expect((JSON.parse(j.cap.out) as { key_source: string }[])[0]!.key_source).toBe("stored");
    });
    it("asks per Galaxy, so a profile for one of two does not mislabel the other", async () => {
      const h = twoLogins();
      h.deps.keySourceFor = (baseUrl) =>
        baseUrl === OTHER
          ? { id: "profile", label: "--profile prod" }
          : { id: "stored", label: "stored login" };
      await statusCommand({ json: true }, h.deps);
      const out = JSON.parse(h.cap.out) as { base_url: string; key_source: string }[];
      expect(out.map((o) => [o.base_url, o.key_source])).toEqual([
        [HOST, "stored"],
        [OTHER, "profile"],
      ]);
    });
  });

  // When the flags or environment send operations to another Galaxy, or an operation would not run, no
  // source is named for the login, and the output says why instead of naming one that would not be used.
  describe("a login that operations would not use", () => {
    const elsewhere = "none (operations go to other.example.org with these flags)";
    it("says so in words, names no source in --json, and does not say something outranks the login", async () => {
      const h = harness({ stored: [fakeLogin()] });
      h.deps.keySourceFor = () => ({ id: null, label: elsewhere });
      expect(await statusCommand({}, h.deps)).toBe(0);
      expect(h.cap.err).toContain(`Key used by operations: ${elsewhere}`);
      expect(h.cap.err).not.toContain("takes precedence");
      const j = harness({ stored: [fakeLogin()] });
      j.deps.keySourceFor = () => ({ id: null, label: elsewhere });
      await statusCommand({ json: true }, j.deps);
      expect((JSON.parse(j.cap.out) as { key_source: string | null }[])[0]!.key_source).toBeNull();
    });
    it("asks with every stored Galaxy, even when --host shows only one", async () => {
      const h = twoLogins();
      const asked: [string, string[]][] = [];
      h.deps.keySourceFor = (baseUrl, _login, stored) => {
        asked.push([baseUrl, [...stored]]);
        return { id: "stored", label: "stored login" };
      };
      await statusCommand({ host: OTHER }, h.deps);
      expect(asked).toEqual([[OTHER, [HOST, OTHER]]]);
    });
  });

  // 77 is EX_NOPERM, as for the other commands that find nothing to use: not logged in.
  it("with nothing stored, says so and exits 77", async () => {
    const h = harness();
    expect(await statusCommand({}, h.deps)).toBe(77);
    expect(h.cap.err).toContain("No stored logins");
  });

  it("with --host and no login for it, says so, names the login command, and exits 77", async () => {
    const h = harness({ stored: [fakeLogin()] });
    expect(await statusCommand({ host: OTHER }, h.deps)).toBe(77);
    expect(h.cap.err).toContain("Not logged in to other.example.org");
    expect(h.cap.err).toContain(`gxy auth login --host ${OTHER}`);
    expect(h.whoamiCalls).toEqual([]);
  });

  // The list leaves out a login a newer gxy wrote; for the host asked for that is not "not logged in".
  it.each([false, true])(
    "with --host naming a login it cannot read, says so and exits 65 (json: %s)",
    async (json) => {
      const h = harness({ stored: [fakeLogin()] });
      h.store.get = async (baseUrl) => {
        if (baseUrl !== OTHER) return null;
        throw new AuthError(
          "store",
          `The login for ${OTHER} was written by a newer version of gxy.`,
        );
      };
      expect(await statusCommand({ host: OTHER, json }, h.deps)).toBe(65);
      expect(h.cap.err).toContain("written by a newer version of gxy");
      expect(h.cap.err).not.toContain("Not logged in");
      expect(h.cap.out).toBe("");
      expect(h.whoamiCalls).toEqual([]);
    },
  );

  // Anything that is not an `AuthError` is a fault in gxy, not something the person can fix.
  it("reports an error it did not expect as unexpected, exit 70", async () => {
    const h = harness({ stored: [fakeLogin()] });
    h.store.list = async () => {
      throw new Error("store is broken");
    };
    expect(await statusCommand({}, h.deps)).toBe(70);
    expect(h.cap.err).toContain("store is broken");
  });

  describe("warnings", () => {
    it("warns for an admin and for a token that does not expire", async () => {
      const h = harness({ stored: [fakeLogin({ method: "pasted", scopes: [], expiresAt: null })] });
      h.knobs.whoami = { username: "root", isAdmin: true };
      await statusCommand({}, h.deps);
      expect(h.cap.err).toContain("full admin rights on dev.galaxy.example.org");
      expect(h.cap.err).toContain("no expiry");
    });
  });

  // Every stored login is checked, so a Galaxy that never answers must cost that one host a timeout
  // and a reported failure, not hang the command or hide the others.
  describe("one slow Galaxy", () => {
    it("is that host's failure, and the others are still reported", async () => {
      const h = twoLogins();
      h.deps.whoami = async (baseUrl) => {
        if (baseUrl !== OTHER) return new Promise(() => {}); // never answers, ignores the signal
        return { username: "bob", isAdmin: false };
      };
      expect(await statusCommand({ timeoutMs: 50 }, h.deps)).toBe(1);
      expect(h.cap.err).toMatch(/No answer from dev\.galaxy\.example\.org within/);
      expect(h.cap.err).toContain("Logged in as bob");
    });

    // Checked one after another, N unreachable Galaxies would cost N timeouts. Neither answers until
    // both have been asked, which a serial loop could never do (it would wait forever on the first).
    it("asks every host at once, and reports them in stored order whichever answers first", async () => {
      const h = twoLogins();
      const asked: string[] = [];
      const answers = new Map<string, (w: { username: string }) => void>();
      h.deps.whoami = (baseUrl) =>
        new Promise((resolve) => {
          asked.push(baseUrl);
          answers.set(baseUrl, resolve);
        });
      const p = statusCommand({}, h.deps);
      await vi.waitFor(() => {
        expect(asked).toHaveLength(2);
      });
      answers.get(OTHER)!({ username: "bob" });
      answers.get(HOST)!({ username: "aioue" });
      expect(await p).toBe(0);
      const err = h.cap.err;
      expect(err.indexOf("dev.galaxy.example.org")).toBeLessThan(err.indexOf("other.example.org"));
      expect(err.indexOf("Logged in as aioue")).toBeLessThan(err.indexOf("Logged in as bob"));
    });

    it("costs one timeout, not one per host", async () => {
      const h = twoLogins();
      h.deps.whoami = () => new Promise(() => {});
      const started = Date.now();
      expect(await statusCommand({ timeoutMs: 400 }, h.deps)).toBe(1);
      expect(Date.now() - started).toBeLessThan(700);
      expect(h.cap.err.match(/No answer from/g)).toHaveLength(2);
    });

    it("stops on Ctrl-C with 130", async () => {
      const h = harness({ stored: [fakeLogin()] });
      h.deps.whoami = () => new Promise(() => {});
      const ac = new AbortController();
      const p = statusCommand({ signal: ac.signal }, h.deps);
      ac.abort();
      expect(await p).toBe(130);
    });
  });

  // `--json` exits 0 once it has a list to print (its option says so): each login's own `status`
  // field carries how its check went, so a failed login is data in the output, not a failed command.
  // A failure before there is a list is still a failure.
  describe("--json", () => {
    it("prints an array to stdout without tokens and exits 0 even when a login failed", async () => {
      const h = harness({ stored: [expiredLogin()] });
      expect(await statusCommand({ json: true }, h.deps)).toBe(0);
      expect(h.cap.err).toBe("");
      const out = JSON.parse(h.cap.out) as Record<string, unknown>[];
      expect(out).toHaveLength(1);
      expect(out[0]).toMatchObject({
        base_url: HOST,
        username: "aioue",
        storage: "keyring",
        method: "device",
        status: "expired",
      });
      expect(h.cap.out).not.toContain(FAKE_TOKEN);
      expect(out[0]).not.toHaveProperty("token");
    });

    it("still fails when there is no list to print, or the user cancels", async () => {
      const broken = harness({ stored: [fakeLogin()] });
      broken.store.list = async () => {
        throw new Error("hosts.yml is not valid");
      };
      expect(await statusCommand({ json: true }, broken.deps)).not.toBe(0);
      expect(broken.cap.out).toBe("");
      const bad = harness({ stored: [fakeLogin()] });
      expect(await statusCommand({ json: true, host: "not a url" }, bad.deps)).not.toBe(0);
      expect(bad.cap.out).toBe("");
      const h = harness({ stored: [fakeLogin()] });
      h.deps.whoami = () => new Promise(() => {});
      const ac = new AbortController();
      const p = statusCommand({ json: true, signal: ac.signal }, h.deps);
      ac.abort();
      expect(await p).toBe(130);
      expect(h.cap.out).toBe("");
    });

    it("is cancelled, not unavailable, when Ctrl-C ends a keyring read", async () => {
      const h = harness({ stored: [fakeLogin()] });
      const ac = new AbortController();
      h.store.get = async () => {
        ac.abort();
        throw new AuthError("not_logged_in", "the keyring helper was ended");
      };
      expect(await statusCommand({ signal: ac.signal }, h.deps)).toBe(130);
      expect(h.cap.err).toContain("Cancelled.");
      expect(h.cap.err).not.toContain("unavailable");
      expect(h.cap.out).toBe("");
    });

    it("includes the token only with --show-token", async () => {
      const h = harness({ stored: [fakeLogin()] });
      await statusCommand({ json: true, showToken: true }, h.deps);
      expect((JSON.parse(h.cap.out) as { token?: string }[])[0]?.token).toBe(FAKE_TOKEN);
    });

    it("carries is_admin from the live check, and null when there was none", async () => {
      const admin = harness({ stored: [fakeLogin()] });
      admin.knobs.whoami = { username: "root", isAdmin: true };
      await statusCommand({ json: true }, admin.deps);
      expect((JSON.parse(admin.cap.out) as { is_admin?: boolean | null }[])[0]?.is_admin).toBe(
        true,
      );

      const unchecked = harness({ stored: [expiredLogin()] });
      await statusCommand({ json: true }, unchecked.deps);
      expect(
        (JSON.parse(unchecked.cap.out) as { is_admin?: boolean | null }[])[0]?.is_admin,
      ).toBeNull();
    });

    it("prints [] when nothing is stored, or nothing is stored for --host", async () => {
      const h = harness();
      expect(await statusCommand({ json: true }, h.deps)).toBe(0);
      expect(JSON.parse(h.cap.out)).toEqual([]);
      const named = harness({ stored: [fakeLogin()] });
      expect(await statusCommand({ json: true, host: OTHER }, named.deps)).toBe(0);
      expect(JSON.parse(named.cap.out)).toEqual([]);
    });
  });

  // The record lists the login but its token cannot be read (the keyring is down, or the secret is
  // gone). That host is reported as unavailable and the others are still checked: one unreadable
  // token must not hide the rest.
  describe("when a token cannot be read", () => {
    it.each(["throw", "null"] as const)(
      "reports that host, still checks the others, and exits 1 (%s)",
      async (how) => {
        const h = twoLogins();
        makeTokenUnreadable(h, [HOST], how);
        expect(await statusCommand({}, h.deps)).toBe(1);
        expect(h.cap.err).toContain("dev.galaxy.example.org");
        expect(h.cap.err).toContain("token unavailable:");
        // No source is named for a token nobody can read, as an operation could not use it either.
        expect(h.cap.err).toContain(
          "Key used by operations: none (the stored token cannot be read)",
        );
        expect(h.cap.err).toContain("Key used by operations: stored login");
        expect(h.cap.err).toContain("other.example.org");
        expect(h.cap.err).toContain("Logged in as"); // the other host was still checked
        expect(h.whoamiCalls.map((c) => c.baseUrl)).toEqual([OTHER]);
      },
    );

    it("gives it a status value in --json, without a token, and exits 0 like any other --json run", async () => {
      const h = twoLogins();
      makeTokenUnreadable(h, [HOST]);
      expect(await statusCommand({ json: true, showToken: true }, h.deps)).toBe(0);
      const out = JSON.parse(h.cap.out) as {
        base_url: string;
        status: string;
        error?: string;
        key_source: string | null;
        token?: string;
      }[];
      expect(out.map((o) => [o.base_url, o.status])).toEqual([
        [HOST, "unavailable"],
        [OTHER, "ok"],
      ]);
      expect(out[0]!.error).toContain("token unavailable:");
      expect(out[0]!.key_source).toBeNull();
      expect(out[1]!.key_source).toBe("stored");
      expect(out[0]!.token).toBeUndefined();
      expect(out[1]!.token).toBe(FAKE_TOKEN);
    });

    it("does the same with --host", async () => {
      const h = twoLogins();
      makeTokenUnreadable(h, [HOST]);
      expect(await statusCommand({ host: HOST }, h.deps)).toBe(1);
      expect(h.cap.err).toContain("token unavailable:");
    });
  });

  // An expiry that cannot be parsed is treated as expired, never as no expiry: a token must not
  // look permanent because its record is damaged.
  it("says an unreadable expiry is unreadable, rather than printing NaN, and makes no live check", async () => {
    const h = harness({ stored: [fakeLogin({ expiresAt: "tomorrow" })] });
    expect(await statusCommand({}, h.deps)).toBe(1);
    expect(h.cap.err).toContain("unreadable expiry");
    expect(h.cap.err).not.toContain("NaN");
    expect(h.whoamiCalls).toEqual([]);
  });
});
