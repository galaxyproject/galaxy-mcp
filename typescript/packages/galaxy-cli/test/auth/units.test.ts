// Small pure modules of the auth surface, one section each.
import { describe, expect, it } from "vitest";
import { abortReason } from "../../src/auth/abort";
import { configDir } from "../../src/auth/paths";
import {
  EXPIRY_MARGIN_MS,
  checkExpiry,
  describeExpiry,
  formatExpiry,
  isExpired,
} from "../../src/auth/expiry";
import { DAY } from "./harness";
import { spawnSync } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";
import { exitQuietlyOnEpipe } from "../../src/auth/epipe";
import { normalizeBaseUrl } from "../../src/auth/base-url";
import { AuthError } from "../../src/auth/types";

// The order is the usual one for a CLI that follows the XDG Base Directory spec (gh does the
// same with its own variable). `env` and `home` are passed in, so no test reads the real ones.
describe("configDir", () => {
  it("prefers GXY_CONFIG_DIR, then XDG_CONFIG_HOME, then ~/.config", () => {
    expect(configDir({ GXY_CONFIG_DIR: "/x", XDG_CONFIG_HOME: "/xdg" }, "/home/u")).toBe("/x");
    expect(configDir({ XDG_CONFIG_HOME: "/xdg" }, "/home/u")).toBe("/xdg/gxy");
    expect(configDir({}, "/home/u")).toBe("/home/u/.config/gxy");
  });
  // The XDG spec says a relative path in `XDG_CONFIG_HOME` is invalid and must be ignored, and a
  // relative path would move the credential file with the working directory, possibly into a
  // repository. The same goes for `GXY_CONFIG_DIR`. An empty one counts as unset.
  it("ignores an empty GXY_CONFIG_DIR and a relative XDG_CONFIG_HOME", () => {
    expect(configDir({ GXY_CONFIG_DIR: "", XDG_CONFIG_HOME: "rel" }, "/home/u")).toBe(
      "/home/u/.config/gxy",
    );
  });
  it("ignores a relative GXY_CONFIG_DIR, falling through to the next in order", () => {
    expect(configDir({ GXY_CONFIG_DIR: "cfg" }, "/home/u")).toBe("/home/u/.config/gxy");
    expect(configDir({ GXY_CONFIG_DIR: "./cfg", XDG_CONFIG_HOME: "/xdg" }, "/home/u")).toBe(
      "/xdg/gxy",
    );
  });
});

// The clock is the harness's fixed `DAY`, never the real one, so a boundary can be tested to the
// millisecond. `at(ms)` is an `expiresAt` that many milliseconds after it.
const now = new Date(DAY);
const at = (ms: number) => new Date(now.getTime() + ms).toISOString();

describe("expiry helper", () => {
  // The offsets are literals and not derived from `EXPIRY_MARGIN_MS`, so moving the margin has to
  // change a test. A token inside the margin is refused because it must still be good for the
  // operation it is handed to; the row at exactly 60 s is the boundary (`<=`).
  it("treats a login as expired 60 s before expiresAt", () => {
    expect(EXPIRY_MARGIN_MS).toBe(60_000);
    expect(isExpired(at(61_000), now)).toBe(false);
    expect(isExpired(at(60_000), now)).toBe(true);
    expect(isExpired(at(30_000), now)).toBe(true);
    expect(isExpired(at(-1), now)).toBe(true);
    expect(isExpired(null, now)).toBe(false);
  });
  // A hand-edited or corrupt `hosts.yml` must not turn a login into one that never expires, which
  // is what treating a bad date like `null` would do.
  it("treats an unparseable expiresAt as expired, never as non-expiring", () => {
    expect(checkExpiry("tomorrow", now)).toEqual({ state: "unreadable" });
    expect(isExpired("tomorrow", now)).toBe(true);
  });
  // `formatExpiry` prints a value from a file a person may have edited, so it must cope with any
  // string and must not pass a terminal escape through.
  it("formatExpiry never throws", () => {
    expect(formatExpiry("tomorrow")).toContain("unreadable");
    expect(formatExpiry("\u001b[2Jx")).not.toContain("\u001b");
    expect(formatExpiry("2026-11-06T18:42:00.000Z")).toBe("2026-11-06 18:42 UTC");
  });
  it("describes the three cases", () => {
    expect(describeExpiry(at(-5000), now)).toMatch(/^expired at /);
    expect(describeExpiry(at(20_000), now)).toMatch(/^is about to expire/);
    expect(describeExpiry("tomorrow", now)).toMatch(/unreadable expiry/);
  });
});

/** The error Node raises on a write to a pipe whose reader has gone. */
const epipe = () => Object.assign(new Error("write EPIPE"), { code: "EPIPE" });
/** Any other stream error, which the handler must not swallow. */
const eio = () => Object.assign(new Error("disk"), { code: "EIO" });

// The streams here are `PassThrough`s and `exit` is a function that records its code, so the
// process under test is never ended. `codes` is everything that would have been passed to it.
describe("exitQuietlyOnEpipe", () => {
  // Exit 0, because the reader leaving (`| head -c1`) is not a failure and the output was wanted.
  // An EPIPE is the only error swallowed: anything else rethrows, or a real fault would vanish.
  it("on stdout exits quietly with 0, and rethrows any other error", () => {
    const stdout = new PassThrough();
    const codes: number[] = [];
    exitQuietlyOnEpipe({ stdout, stderr: new PassThrough() }, (c) => void codes.push(c));
    stdout.emit("error", epipe());
    expect(codes).toEqual([0]);
    expect(() => stdout.emit("error", eio())).toThrow("disk");
  });

  // A closed stderr is different: the login is still running, and exiting 0 would report
  // success for work that has not finished. So the run goes on with stderr silenced, and its own
  // exit code still says how it went. `stderr.read()` being null shows the later write went nowhere.
  it("on stderr does not exit, silences later stderr writes, and rethrows any other error", () => {
    const [stdout, stderr] = [new PassThrough(), new PassThrough()];
    const codes: number[] = [];
    exitQuietlyOnEpipe({ stdout, stderr }, (c) => void codes.push(c));
    expect(() => stderr.emit("error", epipe())).not.toThrow();
    expect(codes).toEqual([]);
    expect(stderr.write("more\n")).toBe(true);
    expect(stderr.read()).toBeNull();
    expect(() => stderr.emit("error", eio())).toThrow("disk");
  });

  // The one test with a real pipe. The tests above inject their streams, so they cannot show that
  // Node really raises EPIPE here and that the process then ends cleanly rather than with an
  // uncaught exception and a stack trace on stderr. `head -c1` closes the pipe after one byte, and
  // the writer sends about 55 MB, far more than a pipe holds, so it must hit the closed pipe.
  // `PIPESTATUS[0]` is the writer's own exit code, which `$?` would replace with head's. It needs
  // bash and head, and Node able to load the module's TypeScript source directly (Node 22.19 or
  // later, which `engines` requires).
  it.skipIf(process.platform === "win32")(
    "a real closed pipe ends the process quietly, with no stack trace",
    async () => {
      const dir = await mkdtemp(join(process.env.GXY_TEST_ROOT ?? tmpdir(), "gxy-epipe-"));
      try {
        const script = join(dir, "writer.mjs");
        const errFile = join(dir, "err");
        await writeFile(
          script,
          `import { exitQuietlyOnEpipe } from ${JSON.stringify(new URL("../../src/auth/epipe.ts", import.meta.url).href)};
exitQuietlyOnEpipe();
const chunk = "fake-token-".repeat(100000);
for (let i = 0; i < 50; i++) process.stdout.write(chunk);
process.stdout.write("\\n");`,
        );
        const r = spawnSync(
          "bash",
          [
            "-c",
            `node ${JSON.stringify(script)} 2>${JSON.stringify(errFile)} | head -c1 >/dev/null; echo \${PIPESTATUS[0]}`,
          ],
          {
            encoding: "utf8",
            timeout: 20_000,
          },
        );
        expect(r.stdout.trim()).toBe("0");
        expect(await readFile(errFile, "utf8")).toBe("");
      } finally {
        await rm(dir, { recursive: true, force: true });
      }
    },
  );
});

// The result is the key every stored login is filed under, so these expectations are written out
// in full: a change to the normal form would silently re-key stored logins, and should fail here.
describe("normalizeBaseUrl", () => {
  it.each([
    ["Host.EXAMPLE", "https://host.example"],
    ["host.example:443", "https://host.example"],
    ["HOST.Example:443/", "https://host.example"],
    ["host.example/", "https://host.example"],
    ["host.example/api", "https://host.example"],
    ["host.example/api/", "https://host.example"],
    ["https://host.example/galaxy/api/", "https://host.example/galaxy"],
    ["host.example/galaxy", "https://host.example/galaxy"],
    ["https://host.example/galaxy//", "https://host.example/galaxy"],
    ["https://host.example/?x=1#frag", "https://host.example"],
    ["  https://host.example  ", "https://host.example"],
    ["http://127.0.0.1:8080/api", "http://127.0.0.1:8080"],
    ["https://bücher.example", "https://xn--bcher-kva.example"],
    ["https://host.example:8443", "https://host.example:8443"],
  ])("%s -> %s", (input, expected) => {
    expect(normalizeBaseUrl(input)).toBe(expected);
  });

  // The result is a credential key and `resolveHost` normalises a stored key again, so it must be a
  // fixed point. A doubled `/api` once normalised to a single `/api`, and that to the bare host, so
  // `token` and `logout` looked for a key that was never stored; a longer run of the idempotence
  // property in properties.test.ts found it, and the examples pin it here. The last row shows
  // that only trailing segments go: a leading `/api` is part of the path.
  it.each([
    ["https://h.example/api/api", "https://h.example"],
    ["https://h.example/api//api/", "https://h.example"],
    ["https://h.example/galaxy/api/api/api", "https://h.example/galaxy"],
    ["https://h.example/api/galaxy/api", "https://h.example/api/galaxy"],
  ])("strips every trailing /api, so normalising twice changes nothing: %s", (input, expected) => {
    const once = normalizeBaseUrl(input);
    expect(once).toBe(expected);
    expect(normalizeBaseUrl(once)).toBe(once);
  });

  // Two paths on one host are two Galaxies (a Galaxy can be served under a prefix), so only an
  // exact `/api` segment is the API root.
  it("keeps only an exact /api suffix: a longer segment is a different Galaxy", () => {
    expect(normalizeBaseUrl("https://host.example/apiary")).toBe("https://host.example/apiary");
    expect(normalizeBaseUrl("https://host.example/api/v2")).toBe("https://host.example/api/v2");
  });

  // Refused, not trimmed: a stray `\r` from a file with Windows line endings would otherwise become
  // a quietly different key. The inputs include U+0085 (a C1 control) and a space inside the host.
  it.each([
    "host.example\r",
    "host.example\n",
    "host.\u001bexample",
    "host.example\u0000",
    "ho st.example",
    "host.example/a b",
    "\u0085host.example",
  ])("refuses control characters and whitespace: %j", (input) => {
    expect(() => normalizeBaseUrl(input)).toThrow(AuthError);
  });

  // A password in the URL would be stored in `hosts.yml`, which is keyed by it.
  it("refuses userinfo, whatever the form", () => {
    expect(() => normalizeBaseUrl("https://user:pw@host.example")).toThrow(/credentials/);
    expect(() => normalizeBaseUrl("user@host.example")).toThrow(/credentials/);
  });

  // The input is never echoed: the credential check used to come after the parse and scheme
  // errors, which printed the whole string, password included.
  it.each([
    "https://user:SECRET@host.example:99999",
    "ftp://user:SECRET@host.example",
    "https://user:SECRET@",
    "user:SECRET@host.example:99999",
    "ftp://user:SECRET@host.example/path?token=SECRET",
  ])("never puts the password in an error: %s", (input) => {
    let message = "";
    try {
      normalizeBaseUrl(input);
    } catch (e) {
      message = (e as Error).message;
    }
    expect(message).not.toBe("");
    expect(message).not.toContain("SECRET");
  });

  // `https:/host` is a typo, and must not become the host "https" with path "/host".
  it.each(["https:/host.example", "http:/localhost", "HTTPS:/host.example/galaxy"])(
    "refuses a scheme with one slash: %s",
    (input) => {
      expect(() => normalizeBaseUrl(input)).toThrow(expect.objectContaining({ kind: "usage" }));
    },
  );

  it("still takes host:port, with or without a path, as a bare host", () => {
    expect(normalizeBaseUrl("host.example:8443")).toBe("https://host.example:8443");
    expect(normalizeBaseUrl("host.example:8443/galaxy")).toBe("https://host.example:8443/galaxy");
  });

  it("refuses plain http except to loopback, and other schemes", () => {
    expect(() => normalizeBaseUrl("http://host.example")).toThrow(
      expect.objectContaining({ kind: "insecure" }),
    );
    expect(() => normalizeBaseUrl("ftp://host.example")).toThrow(AuthError);
  });

  // The gate that decides whether a token may cross the network in clear text, so both sides are
  // pinned: what it accepts here, and (next) the hosts that only look like loopback. Dropping the
  // `$` of the 127.x pattern, for one, would send a token over http to
  // `127.0.0.1.evil.example`.
  it.each([
    ["http://localhost", "http://localhost"],
    ["http://localhost:8080", "http://localhost:8080"],
    ["http://[::1]:8080", "http://[::1]:8080"],
    ["http://127.0.0.1", "http://127.0.0.1"],
    ["http://127.1.2.3", "http://127.1.2.3"],
  ])("accepts plain http to loopback: %s", (input, expected) => {
    expect(normalizeBaseUrl(input)).toBe(expected);
  });

  // Each looks like loopback to a careless check and is not: a name an attacker could register, a
  // path that contains one, a neighbouring address (`[::2]`, and `0.0.0.0`, which is the
  // unspecified address), or userinfo in front of another host.
  it.each([
    ["http://127.0.0.1.evil.example", "insecure"],
    ["http://localhost.evil.example", "insecure"],
    ["http://evil.example/127.0.0.1", "insecure"],
    ["http://[::2]", "insecure"],
    ["http://0.0.0.0", "insecure"],
    ["http://127.0.0.1@evil.example", "usage"], // refused for the userinfo first, refused either way
  ])("refuses plain http to a host that only looks like loopback: %s", (input, kind) => {
    expect(() => normalizeBaseUrl(input)).toThrow(expect.objectContaining({ kind }));
  });
});

describe("abortReason", () => {
  it("passes an Error reason through as it is, so callers can still match on it", () => {
    const reason = new Error("ctrl-c");
    const controller = new AbortController();
    controller.abort(reason);
    expect(abortReason(controller.signal)).toBe(reason);
  });

  it("gives the AbortError of a plain abort()", () => {
    const controller = new AbortController();
    controller.abort();
    expect(abortReason(controller.signal)).toBeInstanceOf(Error);
    expect(abortReason(controller.signal).name).toBe("AbortError");
  });

  it("wraps a reason that is not an Error, keeping it as the cause", () => {
    const controller = new AbortController();
    controller.abort("stop");
    const error = abortReason(controller.signal);
    expect(error).toBeInstanceOf(Error);
    expect(error.cause).toBe("stop");
  });
});
