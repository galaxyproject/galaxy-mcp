/**
 * `login` as a command, in process: fakes for the store and the terminal, so every branch can be
 * reached and each run's output read back.
 */
import { describe, it, expect, vi } from "vitest";
import { loginCommand } from "../../src/auth/commands/login";
import { markAnswered } from "../../src/auth/network";
import { createWhoami } from "../../src/auth/whoami";
import { AuthError } from "../../src/auth/types";
import { FAKE_TOKEN, HOST, harness } from "./harness";

const HOST_ARG = "https://dev.galaxy.example.org";
const never = <T>() => new Promise<T>(() => {});

// The host is shown before a key is read, so the person sees where it is going while they can
// still stop it (login.ts).
describe("naming the Galaxy", () => {
  /** A harness whose stdin holds a key, for a login that gets as far as reading one. */
  const withKey = (opts: Parameters<typeof harness>[0] = {}) =>
    harness({ stdinText: `${FAKE_TOKEN}\n`, ...opts });

  it("shows the host before a pasted key is read or sent", async () => {
    const h = withKey();
    const seen: string[] = [];
    h.deps.whoami = async () => {
      seen.push(h.cap.err);
      return { username: "aioue" };
    };
    await loginCommand({ host: HOST, withToken: true }, h.deps);
    expect(seen[0]).toContain("Logging in to dev.galaxy.example.org");
  });

  it("is a usage error without a host when not interactive", async () => {
    const h = harness();
    expect(await loginCommand({ withToken: true }, h.deps)).toBe(64);
    expect(h.cap.err).toContain("--host");
  });

  // Login never takes its host from `.env` or a profile (`pickHost`): a repository must not choose where
  // your credentials go. When one names a different Galaxy, login says so, before "Logging in to",
  // so the person is not left wondering why it went elsewhere.
  it("says so when .env names a different Galaxy than the one it uses", async () => {
    const h = withKey({
      hostFromEnv: HOST_ARG,
      ignoredLoginHost: { source: "dotenv", url: "https://other.example.org" },
    });
    expect(await loginCommand({ withToken: true }, h.deps)).toBe(0);
    const note = h.cap.err.indexOf("Ignoring GALAXY_URL from .env (https://other.example.org)");
    expect(note).toBeGreaterThanOrEqual(0);
    expect(note).toBeLessThan(h.cap.err.indexOf("Logging in to"));
  });

  it.each([
    [
      "when .env names the chosen host, even written differently",
      {
        hostFromEnv: HOST_ARG,
        ignoredLoginHost: { source: "dotenv" as const, url: `${HOST_ARG}/` },
      },
      {},
    ],
    [
      "with --host, which is named on purpose",
      { ignoredLoginHost: { source: "dotenv" as const, url: "https://other.example.org" } },
      { host: HOST_ARG },
    ],
  ])("stays quiet about .env %s", async (_when, setup, opts) => {
    const h = withKey(setup);
    await loginCommand({ withToken: true, ...opts }, h.deps);
    expect(h.cap.err).not.toContain("Ignoring");
  });
});

// The caller hands `loginCommand` an `AbortSignal` (commands do not touch process signals), and
// every step either takes it or is raced against it. Each test cancels at a different step, since
// each is a different place to be left holding something: a terminal in raw mode, a key about to be
// written. 130 is 128 plus SIGINT's 2, the shell's code for a Ctrl-C.
describe("cancelling with Ctrl-C", () => {
  const expectCancelled = (h: ReturnType<typeof harness>, code: number) => {
    expect(code).toBe(130);
    expect(h.cap.err).toContain("Cancelled");
    expect(h.store.putCalls).toEqual([]);
    expect(h.cap.out).toBe("");
  };

  it("does nothing when the signal is already aborted", async () => {
    const h = harness({ stdinText: `${FAKE_TOKEN}\n` });
    const ac = new AbortController();
    ac.abort();
    expectCancelled(
      h,
      await loginCommand({ host: HOST_ARG, withToken: true, signal: ac.signal }, h.deps),
    );
    expect(h.whoamiCalls).toEqual([]);
  });

  it("during whoami, which is told to stop, and stores nothing", async () => {
    const h = harness({ stdinText: `${FAKE_TOKEN}\n` });
    const ac = new AbortController();
    let seen: AbortSignal | undefined;
    h.deps.whoami = (_b, _t, signal) => ((seen = signal), never());
    const p = loginCommand({ host: HOST_ARG, withToken: true, signal: ac.signal }, h.deps);
    await vi.waitFor(() => {
      expect(seen).toBeDefined();
    });
    ac.abort();
    expectCancelled(h, await p);
    expect(seen!.aborted).toBe(true);
  });

  it("without writing the store when the cancel lands as whoami answers", async () => {
    const h = harness({ stdinText: `${FAKE_TOKEN}\n` });
    const ac = new AbortController();
    h.deps.whoami = async () => {
      ac.abort();
      return { username: "aioue" };
    };
    expectCancelled(
      h,
      await loginCommand({ host: HOST_ARG, withToken: true, signal: ac.signal }, h.deps),
    );
  });

  it("while reading a piped key, before any whoami, leaving no listener on stdin", async () => {
    const h = harness(); // stdin stays open and empty
    const ac = new AbortController();
    const p = loginCommand({ host: HOST_ARG, withToken: true, signal: ac.signal }, h.deps);
    await vi.waitFor(() => {
      expect(h.cap.err).toContain("Logging in to");
    });
    ac.abort();
    expectCancelled(h, await p);
    expect(h.whoamiCalls).toEqual([]);
    expect(h.stdin.listenerCount("data")).toBe(0);
  });
});

describe("login --with-token", () => {
  it("refuses without --with-token, naming the program and the way in, and reads nothing", async () => {
    const h = harness({ stdinText: `${FAKE_TOKEN}\n` });
    expect(await loginCommand({ host: HOST_ARG }, h.deps)).toBe(64);
    expect(h.cap.err).toContain("gxy auth login needs --with-token");
    expect(h.cap.err).toContain('printf %s "$KEY" | gxy auth login --host <url> --with-token');
    expect(h.cap.err).not.toContain("Logging in to");
    expect(h.whoamiCalls).toEqual([]);
    expect(h.store.putCalls).toEqual([]);
  });

  it("reads the key from stdin, trims it, and stores a pasted login", async () => {
    const h = harness({ stdinText: `  ${FAKE_TOKEN}\n` });
    expect(await loginCommand({ host: HOST_ARG, withToken: true }, h.deps)).toBe(0);
    expect(h.whoamiCalls).toEqual([{ baseUrl: HOST, token: FAKE_TOKEN }]);
    expect(h.store.putCalls[0]!.login).toMatchObject({
      method: "pasted",
      scopes: [],
      expiresAt: null,
      token: FAKE_TOKEN,
      username: "aioue",
    });
    expect(h.cap.err).toContain("Logged in to dev.galaxy.example.org as aioue");
    expect(h.cap.out).toBe("");
  });

  it("says where the token went", async () => {
    const h = harness({ stdinText: `${FAKE_TOKEN}\n` });
    expect(await loginCommand({ host: HOST_ARG, withToken: true }, h.deps)).toBe(0);
    expect(h.cap.err).toContain("Token stored in a file readable only by you (no expiry)");
    expect(h.store.logins.get(HOST)?.storage).toBe("file");
  });

  it("with --insecure-storage, changes nothing for an ordinary key", async () => {
    const h = harness({ stdinText: `${FAKE_TOKEN}\n` });
    expect(
      await loginCommand({ host: HOST_ARG, withToken: true, insecureStorage: true }, h.deps),
    ).toBe(0);
    expect(h.store.logins.get(HOST)?.token).toBe(FAKE_TOKEN);
  });

  it.each([
    ["empty", ""],
    ["blank", "  \n"],
    ["multi-line", "abc\ndef\n"],
  ])("refuses %s input", async (_n, text) => {
    const h = harness({ stdinText: text });
    expect(await loginCommand({ host: HOST_ARG, withToken: true }, h.deps)).toBe(64);
    expect(h.store.putCalls).toEqual([]);
    expect(h.cap.err).not.toContain("abc");
  });

  it("exits 77 and stores nothing when Galaxy rejects the key", async () => {
    const h = harness({ stdinText: FAKE_TOKEN });
    h.knobs.whoami = new AuthError("not_logged_in", "Galaxy rejected the key.");
    expect(await loginCommand({ host: HOST_ARG, withToken: true }, h.deps)).toBe(77);
    expect(h.store.putCalls).toEqual([]);
  });

  // The file is the person's to fix, not a bug in gxy: the message as it is, and 65, not "Unexpected error" and 70.
  it("reports a store that refuses its file with that message and 65, not as an unexpected error", async () => {
    const h = harness({ stdinText: FAKE_TOKEN });
    h.store.putError = new AuthError(
      "store",
      "Refusing to use /home/x/.config/gxy/hosts.yml: it is a symbolic link. Remove it or point GXY_CONFIG_DIR somewhere else.",
    );
    expect(await loginCommand({ host: HOST_ARG, withToken: true }, h.deps)).toBe(65);
    expect(h.cap.err).toContain("Refusing to use /home/x/.config/gxy/hosts.yml");
    expect(h.cap.err).not.toContain("Unexpected error");
    expect(h.cap.err).not.toContain(FAKE_TOKEN);
  });

  it("names the certificate setting when Galaxy cannot be reached securely", async () => {
    const h = harness({ stdinText: FAKE_TOKEN });
    const tls = (async () => {
      throw new TypeError("fetch failed", {
        cause: Object.assign(new Error("self-signed certificate"), {
          code: "DEPTH_ZERO_SELF_SIGNED_CERT",
        }),
      });
    }) as unknown as typeof fetch;
    h.deps.whoami = createWhoami(tls);
    expect(await loginCommand({ host: HOST_ARG, withToken: true }, h.deps)).toBe(69);
    expect(h.cap.err).toContain("NODE_EXTRA_CA_CERTS");
    expect(h.store.putCalls).toEqual([]);
  });

  describe("proxy hint on a network failure", () => {
    const HINT = "Node ignores it unless NODE_USE_ENV_PROXY=1 is also set";
    const failWith = async (
      env: Record<string, string>,
      err = new AuthError("network", "Could not reach it."),
    ) => {
      const h = harness({ env, stdinText: FAKE_TOKEN });
      h.deps.whoami = async () => {
        throw err;
      };
      const code = await loginCommand({ host: HOST_ARG, withToken: true }, h.deps);
      return { code, err: h.cap.err, out: h.cap.out };
    };

    it.each(["HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY"])(
      "names %s when it is set without NODE_USE_ENV_PROXY",
      async (name) => {
        const r = await failWith({ [name]: "http://proxy.example.org:3128" });
        expect(r.code).toBe(69);
        expect(r.err).toContain(`${name} is set, but ${HINT} (Node 22.21+ or 24+).`);
      },
    );

    it("is absent when NODE_USE_ENV_PROXY=1", async () => {
      const r = await failWith({
        HTTPS_PROXY: "http://proxy.example.org:3128",
        NODE_USE_ENV_PROXY: "1",
      });
      expect(r.code).toBe(69);
      expect(r.err).not.toContain(HINT);
    });

    it("is absent when no proxy variable is set", async () => {
      const r = await failWith({});
      expect(r.code).toBe(69);
      expect(r.err).not.toContain(HINT);
    });

    // The server answered (a 5xx, a 429), so an ignored proxy is not the cause.
    it("is absent when the server answered", async () => {
      const r = await failWith(
        { HTTPS_PROXY: "http://proxy.example.org:3128" },
        markAnswered(new AuthError("network", "https://x answered HTTP 503.")),
      );
      expect(r.code).toBe(69);
      expect(r.err).not.toContain(HINT);
    });

    it("is absent for an error that is not a network error", async () => {
      const r = await failWith(
        { HTTPS_PROXY: "http://proxy.example.org:3128" },
        new AuthError("not_logged_in", "Galaxy rejected the key."),
      );
      expect(r.code).toBe(77);
      expect(r.err).not.toContain(HINT);
    });

    it("never prints the proxy URL or its credentials", async () => {
      const r = await failWith({ HTTPS_PROXY: "http://user:secret@proxy.example.org:3128" });
      expect(r.err).toContain(HINT);
      for (const text of [r.err, r.out]) {
        expect(text).not.toContain("secret");
        expect(text).not.toContain("user:");
        expect(text).not.toContain("proxy.example.org");
      }
    });
  });

  describe("on a pipe", () => {
    it("does not prompt, even when otherwise interactive", async () => {
      const h = harness({ interactive: true, stdinText: FAKE_TOKEN });
      expect(await loginCommand({ host: HOST_ARG, withToken: true }, h.deps)).toBe(0);
      expect(h.cap.err).not.toContain("Paste your Galaxy API key");
    });

    // An API key is tens of characters (stdin.ts), so a pipe that is far larger is not one, and is
    // refused before anything is sent to Galaxy.
    it("refuses more than 16 KiB with a usage error, before any whoami, without echoing it", async () => {
      const h = harness({ stdinText: "k".repeat(16 * 1024 + 1) });
      expect(await loginCommand({ host: HOST_ARG, withToken: true }, h.deps)).toBe(64);
      expect(h.cap.err).toContain("16 KiB");
      expect(h.whoamiCalls).toEqual([]);
      expect(h.cap.err).not.toContain("kkkk");
    });
  });

  describe("at a terminal", () => {
    /**
     * A stdin that looks like a TTY and records raw mode. The terminal is restored when `raw` ends on
     * false: raw mode (no echo, no line editing) outlasts the process if it is left on.
     */
    function terminal() {
      const h = harness({ interactive: true });
      const raw: boolean[] = [];
      Object.assign(h.stdin, { isTTY: true, setRawMode: (m: boolean) => void raw.push(m) });
      return {
        h,
        raw,
        prompted: () =>
          vi.waitFor(() => {
            expect(h.cap.err).toContain("Paste your Galaxy API key");
          }),
      };
    }

    it("prompts on stderr, reads without echo, and restores the terminal", async () => {
      const { h, raw, prompted } = terminal();
      const p = loginCommand({ host: HOST_ARG, withToken: true }, h.deps);
      await prompted();
      h.stdin.write(`${FAKE_TOKEN}\r`);
      expect(await p).toBe(0);
      expect(raw.at(-1)).toBe(false);
      expect(h.whoamiCalls).toEqual([{ baseUrl: HOST, token: FAKE_TOKEN }]);
      expect(h.cap.err + h.cap.out).not.toContain(FAKE_TOKEN);
      expect(h.cap.out).toBe("");
    });

    // In raw mode Ctrl-C arrives as the character \x03, not as SIGINT, so the prompt has to cancel itself.
    it("cancels on Ctrl-C at the prompt: exit 130, nothing stored, terminal restored", async () => {
      const { h, raw, prompted } = terminal();
      const p = loginCommand({ host: HOST_ARG, withToken: true }, h.deps);
      await prompted();
      h.stdin.write("fake\x03");
      expect(await p).toBe(130);
      expect(raw.at(-1)).toBe(false);
      expect(h.cap.err).toContain("Cancelled.");
      expect(h.whoamiCalls).toEqual([]);
      expect(h.store.putCalls).toEqual([]);
    });

    it("restores the terminal when the abort signal fires instead", async () => {
      const { h, raw, prompted } = terminal();
      const ac = new AbortController();
      const p = loginCommand({ host: HOST_ARG, withToken: true, signal: ac.signal }, h.deps);
      await prompted();
      ac.abort();
      expect(await p).toBe(130);
      expect(raw.at(-1)).toBe(false);
    });

    it("restores the terminal when the key is refused", async () => {
      const { h, raw, prompted } = terminal();
      h.knobs.whoami = new AuthError("not_logged_in", "Galaxy rejected the key.");
      const p = loginCommand({ host: HOST_ARG, withToken: true }, h.deps);
      await prompted();
      h.stdin.write("fake-key\n");
      expect(await p).toBe(77);
      expect(raw.at(-1)).toBe(false);
    });

    // Ctrl-D, or Enter, on an empty prompt: "on stdin" would be the wrong place to say it was wanted.
    it("says no key was entered, not that stdin had none, when the prompt is left empty", async () => {
      const { h, raw, prompted } = terminal();
      const p = loginCommand({ host: HOST_ARG, withToken: true }, h.deps);
      await prompted();
      h.stdin.write("\x04");
      expect(await p).toBe(64);
      expect(raw.at(-1)).toBe(false);
      expect(h.cap.err).toContain("No API key entered");
      expect(h.cap.err).not.toContain("on stdin");
      expect(h.whoamiCalls).toEqual([]);
    });

    // The prompt goes to stderr. With stderr redirected (`2>log`, `| tee`) nobody sees it, and
    // reading stdin as a pipe would echo the typed key in cooked mode: so it is refused, not read.
    it("refuses, reading nothing, when stderr is not a terminal", async () => {
      const h = harness({ interactive: false });
      const raw: boolean[] = [];
      Object.assign(h.stdin, { isTTY: true, setRawMode: (m: boolean) => void raw.push(m) });
      h.stdin.write(`${FAKE_TOKEN}\n`);
      expect(await loginCommand({ host: HOST_ARG, withToken: true }, h.deps)).toBe(64);
      expect(raw).toEqual([]);
      expect(h.cap.err).toContain("stdin is a terminal but stderr is not");
      expect(h.cap.err).toContain(`printf %s "$KEY" | gxy auth login --host ${HOST}`);
      expect(h.cap.err).not.toContain("Paste your Galaxy API key");
      expect(h.cap.err + h.cap.out).not.toContain(FAKE_TOKEN);
      expect(h.whoamiCalls).toEqual([]);
    });
  });
});

// An admin key is the whole server, and the only place a login is kept is a plaintext file, so
// login refuses it with 77, after the warning, unless `--insecure-storage` asks for the file.
describe("tokens that carry admin rights", () => {
  const admin = { username: "root", isAdmin: true };

  it("get a warning, printed once, and the login is stored when --insecure-storage says so", async () => {
    const h = harness({ stdinText: `${FAKE_TOKEN}\n` });
    h.knobs.whoami = admin;
    expect(
      await loginCommand({ host: HOST_ARG, withToken: true, insecureStorage: true }, h.deps),
    ).toBe(0);
    expect(h.cap.err.match(/full admin rights on dev\.galaxy\.example\.org/g)).toHaveLength(1);
    expect(h.store.logins.get(HOST)?.token).toBe(FAKE_TOKEN);
  });

  it("do not make an ordinary user's login warn", async () => {
    const h = harness({ stdinText: `${FAKE_TOKEN}\n` });
    await loginCommand({ host: HOST_ARG, withToken: true }, h.deps);
    expect(h.cap.err).not.toContain("admin rights");
  });

  it("are refused with 77, after the warning, unless --insecure-storage says so", async () => {
    const h = harness({ stdinText: `${FAKE_TOKEN}\n` });
    h.knobs.whoami = admin;
    expect(await loginCommand({ host: HOST_ARG, withToken: true }, h.deps)).toBe(77);
    expect(h.store.putCalls).toEqual([]);
    expect(h.store.logins.size).toBe(0);
    const warned = h.cap.err.indexOf("full admin rights");
    expect(warned).toBeGreaterThanOrEqual(0);
    // The refusal names the way out and the alternative that stores nothing.
    const refused = h.cap.err.search(/--insecure-storage.*GALAXY_API_KEY/s);
    expect(refused).toBeGreaterThan(warned);
    expect(h.cap.err).not.toContain("Logged in");
    expect(h.cap.err).not.toContain(FAKE_TOKEN);
  });

  it("add their warning to a pasted key's own no-expiry warning", async () => {
    const plain = harness({ stdinText: `${FAKE_TOKEN}\n` });
    await loginCommand({ host: HOST_ARG, withToken: true }, plain.deps);
    expect(plain.cap.err).toContain("no expiry");
    expect(plain.cap.err).not.toContain("admin rights");

    const root = harness({ stdinText: `${FAKE_TOKEN}\n` });
    root.knobs.whoami = admin;
    await loginCommand({ host: HOST_ARG, withToken: true, insecureStorage: true }, root.deps);
    expect(root.cap.err).toContain("admin rights");
    expect(root.cap.err).toContain("no expiry");
  });
});
