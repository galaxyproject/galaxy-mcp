/**
 * `login` as a command, in process: fakes for the server, the store and the terminal, so every
 * branch can be reached and each run's output read back. The one block that uses real discovery
 * says so. The real binary, a real SIGINT and real terminals are in e2e.int.test.ts.
 */
import { readFile } from "node:fs/promises";
import { describe, it, expect, vi } from "vitest";
import { loginCommand } from "../../src/auth/commands/login";
import { discover } from "../../src/auth/oauth";
import { markAnswered } from "../../src/auth/network";
import { createWhoami } from "../../src/auth/whoami";
import { AuthError, type BaseUrl } from "../../src/auth/types";
import { useFakeGalaxies, type MetadataMode } from "../support/fake-galaxy";
import {
  FAKE_TOKEN,
  HOST,
  REFUSED_WITHOUT_KEYRING,
  FakeKeyring,
  harness,
  realStore,
  useTempDir,
} from "./harness";

const HOST_ARG = "https://dev.galaxy.example.org";
const never = <T>() => new Promise<T>(() => {});

describe("login (device flow)", () => {
  it("shows the code and page on stderr, stores the login, and keeps stdout empty", async () => {
    const h = harness();
    const code = await loginCommand(
      { host: HOST_ARG, scopes: ["histories:read", "tools:run"] },
      h.deps,
    );
    expect(code).toBe(0);
    expect(h.cap.out).toBe("");
    expect(h.cap.err).toContain("WDJB-MJHT");
    expect(h.cap.err).toContain("https://dev.galaxy.example.org/login/device");
    // RFC 8628 §5.4 (remote phishing): anyone can start a login and ask someone else to approve its code.
    expect(h.cap.err).toMatch(/only approve this code/i);
    expect(h.cap.err).toContain("Logged in to dev.galaxy.example.org as aioue");
    expect(h.cap.err).toMatch(/stored in the system keyring/);
    expect(h.server.startedWith).toEqual([["histories:read", "tools:run"]]);
    expect(h.server.revoked).toEqual([]); // a good login keeps its token
    expect(h.store.putCalls[0]!.login).toMatchObject({
      baseUrl: HOST,
      username: "aioue",
      method: "device",
      scopes: ["histories:read"],
      token: FAKE_TOKEN,
      expiresAt: "2026-11-06T18:42:00.000Z",
    });
  });

  it("asks for no scopes when --scopes is not given", async () => {
    const h = harness();
    await loginCommand({ host: HOST_ARG }, h.deps);
    expect(h.server.startedWith).toEqual([undefined]);
  });

  it("stores no expiry when the server gave none, and relays the store's warning", async () => {
    const h = harness();
    h.server.pending.outcome = { accessToken: FAKE_TOKEN, scopes: [] };
    h.store.nextStorage = "file";
    h.store.nextWarning = "No keyring here; the token went to a file.";
    await loginCommand({ host: HOST_ARG }, h.deps);
    expect(h.store.putCalls[0]!.login.expiresAt).toBeNull();
    expect(h.cap.err).toContain("No keyring here; the token went to a file.");
    expect(h.cap.err).toMatch(/stored in a file/);
    expect(h.cap.err).toMatch(/no expiry/);
  });

  it("with --insecure-storage, keeps the token in the file", async () => {
    const h = harness();
    await loginCommand({ host: HOST_ARG, insecureStorage: true }, h.deps);
    expect(h.store.logins.get(HOST)?.storage).toBe("file");
  });

  // An explicit key beats a stored login for operations (config.ts), so a login stored now would
  // quietly go unused. The warning says so, and never repeats the value.
  it("warns that GALAXY_API_KEY takes precedence, without printing it", async () => {
    const h = harness({ env: { GALAXY_API_KEY: "irrelevant" } });
    await loginCommand({ host: HOST_ARG }, h.deps);
    expect(h.cap.err).toContain("GALAXY_API_KEY is set and will take precedence");
    expect(h.cap.err).not.toContain("irrelevant");
  });

  describe("the browser", () => {
    it("is offered, and opens the approval page on Enter, when interactive", async () => {
      const h = harness({ interactive: true });
      h.server.pending.outcome = "hang";
      const ac = new AbortController();
      const p = loginCommand({ host: HOST_ARG, signal: ac.signal }, h.deps);
      await vi.waitFor(() => {
        expect(h.cap.err).toContain("Waiting for approval");
      });
      expect(h.cap.err).toContain("Press Enter");
      expect(h.opened).toEqual([]);
      h.stdin.write("\n");
      await vi.waitFor(() => {
        expect(h.opened).toEqual(["https://dev.galaxy.example.org/login/device"]);
      });
      ac.abort();
      expect(await p).toBe(130);
    });

    // Interactive means stdin and stderr are both terminals. Anything else (a pipe, a file) must
    // not wait for Enter, and e2e.int.test.ts shows that on real streams.
    it.each([
      ["not interactive", { interactive: false }, {}],
      ["--no-browser is given", { interactive: true }, { browser: false }],
      ["no browser is available", { interactive: true, noBrowser: true }, {}],
    ] as const)("is not offered when %s", async (_when, setup, opts) => {
      const h = harness({ interactive: setup.interactive });
      if ("noBrowser" in setup) h.knobs.browserAvailable = false;
      expect(await loginCommand({ host: HOST_ARG, ...opts }, h.deps)).toBe(0);
      expect(h.cap.err).not.toContain("Press Enter");
      expect(h.opened).toEqual([]);
    });
  });

  // Once the server has issued a token, one that cannot be checked or kept would stay valid (an
  // admin's too) until it expires, with nothing local pointing at it. So login takes it back, and
  // says what happened either way, since the person may have to revoke it by hand.
  describe("when something goes wrong", () => {
    it("stores nothing, exits 77 and takes the token back when Galaxy refuses it", async () => {
      const h = harness();
      h.knobs.whoami = new AuthError("not_logged_in", "Galaxy rejected the token.");
      expect(await loginCommand({ host: HOST_ARG }, h.deps)).toBe(77);
      expect(h.store.putCalls).toEqual([]);
      expect(h.server.revoked).toEqual([FAKE_TOKEN]);
    });

    // 77 is EX_NOPERM and 69 EX_UNAVAILABLE (exit.ts): being refused and not being able to ask stay
    // different codes.
    it.each([
      ["denied", 77],
      ["network", 69],
    ] as const)("exits %s -> %i, relays the message and stores nothing", async (kind, exit) => {
      const h = harness();
      h.server.pending.outcome = new AuthError(kind, `it was ${kind}`);
      expect(await loginCommand({ host: HOST_ARG }, h.deps)).toBe(exit);
      expect(h.cap.err).toContain(`it was ${kind}`);
      expect(h.store.putCalls).toEqual([]);
    });

    it("revokes the token it was just given when the login cannot be stored", async () => {
      const h = harness();
      h.store.putError = new Error("disk full");
      expect(await loginCommand({ host: HOST_ARG }, h.deps)).not.toBe(0);
      expect(h.server.revoked).toEqual([FAKE_TOKEN]);
      expect(h.cap.err).toContain("so it was revoked");
      expect(h.cap.err + h.cap.out).not.toContain(FAKE_TOKEN);
    });

    it("revokes the token when whoami fails unexpectedly", async () => {
      const h = harness();
      h.deps.whoami = async () => {
        throw new Error("whoami exploded");
      };
      expect(await loginCommand({ host: HOST_ARG }, h.deps)).not.toBe(0);
      expect(h.server.revoked).toEqual([FAKE_TOKEN]);
      expect(h.cap.err).toContain("so it was revoked");
      expect(h.store.putCalls).toEqual([]);
    });

    it("revokes the token of an admin who is refused for lack of a keyring, without echoing it", async () => {
      const h = harness();
      h.store.keyringAvailable = false;
      h.knobs.whoami = { username: "root", isAdmin: true };
      expect(await loginCommand({ host: HOST_ARG }, h.deps)).toBe(77);
      expect(h.server.revoked).toEqual([FAKE_TOKEN]);
      expect(h.cap.err).toContain("so it was revoked");
      expect(h.cap.err + h.cap.out).not.toContain(FAKE_TOKEN);
    });

    it("says to revoke it by hand when that revocation fails too", async () => {
      const h = harness();
      h.store.putError = new Error("disk full");
      h.server.revokeError = new Error("no route");
      expect(await loginCommand({ host: HOST_ARG }, h.deps)).not.toBe(0);
      expect(h.cap.err).toContain("revoking it failed");
      expect(h.cap.err).toContain("user preferences");
    });

    it("says to revoke it by hand when the server has no revocation endpoint", async () => {
      const h = harness();
      h.store.putError = new Error("disk full");
      h.server.supportsRevocation = false;
      expect(await loginCommand({ host: HOST_ARG }, h.deps)).not.toBe(0);
      expect(h.server.revoked).toEqual([]);
      expect(h.cap.err).toContain("cannot be revoked from here");
    });
  });

  // The host is shown before a key is read or a code is requested, so the person sees where it is
  // going while they can still stop it (login.ts).
  describe("naming the Galaxy", () => {
    it("shows the host before a pasted key is read or sent", async () => {
      const h = harness({ stdinText: `${FAKE_TOKEN}\n` });
      const seen: string[] = [];
      h.deps.whoami = async () => {
        seen.push(h.cap.err);
        return { username: "aioue" };
      };
      await loginCommand({ host: HOST, withToken: true }, h.deps);
      expect(seen[0]).toContain("Logging in to dev.galaxy.example.org");
    });

    it("shows the host before the device flow starts", async () => {
      const h = harness();
      const seen: string[] = [];
      h.deps.discover = async () => {
        seen.push(h.cap.err);
        return h.server;
      };
      await loginCommand({ host: HOST }, h.deps);
      expect(seen[0]).toContain("Logging in to dev.galaxy.example.org");
    });

    it("is a usage error without a host when not interactive", async () => {
      const h = harness();
      expect(await loginCommand({}, h.deps)).toBe(64);
      expect(h.cap.err).toContain("--host");
    });

    it("prompts for the URL when interactive and nothing names one", async () => {
      const h = harness({ interactive: true });
      h.stdin.write("dev.galaxy.example.org\n");
      expect(await loginCommand({ browser: false }, h.deps)).toBe(0);
      expect(h.cap.err).toContain("Galaxy URL");
      expect(h.store.putCalls[0]!.login.baseUrl).toBe(HOST);
    });

    // Login never takes its host from `.env` or a profile (`pickHost`): a repository must not choose where
    // your credentials go. When one names a different Galaxy, login says so, before "Logging in to",
    // so the person is not left wondering why it went elsewhere.
    it("says so when .env names a different Galaxy than the one it uses", async () => {
      const h = harness({
        hostFromEnv: HOST_ARG,
        ignoredLoginHost: { source: "dotenv", url: "https://other.example.org" },
      });
      expect(await loginCommand({}, h.deps)).toBe(0);
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
      const h = harness(setup);
      await loginCommand(opts, h.deps);
      expect(h.cap.err).not.toContain("Ignoring");
    });
  });
});

// A Galaxy can lack device login in three ways: a 404 for its metadata, a page that is not JSON, or
// metadata with no device endpoint. All three get one message and exit 76, EX_PROTOCOL, with the
// hint to paste a key instead. Broken metadata and metadata for another issuer are different
// faults, and get no such hint.
describe("login against a Galaxy with no device login", () => {
  const galaxyOf = useFakeGalaxies();

  /** Real discovery against the fake, so the wording follows what Galaxy actually sends. */
  async function login(mode: MetadataMode | "no-device") {
    const galaxy = await galaxyOf();
    if (mode === "no-device") galaxy.setDeviceFlow(false);
    else galaxy.setMetadataMode(mode);
    const h = harness();
    h.deps.discover = (u: BaseUrl, signal?: AbortSignal) => discover(u, { signal });
    const code = await loginCommand({ host: galaxy.baseUrl }, h.deps);
    return { code, err: h.cap.err, hint: `gxy auth login --host ${galaxy.baseUrl} --with-token` };
  }

  it.each(["not-found", "not-json", "no-device"] as const)(
    "exits 76 with the --with-token hint (%s)",
    async (mode) => {
      const { code, err, hint } = await login(mode);
      expect(code).toBe(76);
      expect(err).toContain("does not offer device login");
      expect(err).toContain(`Paste an API key instead: ${hint}`);
    },
  );

  it("does not offer the hint for malformed metadata, which is a protocol problem", async () => {
    const { code, err } = await login("malformed");
    expect(code).toBe(76);
    expect(err).not.toContain("does not offer device login");
  });

  it("calls metadata for another issuer insecure, without the hint", async () => {
    const { code, err } = await login("wrong-issuer");
    expect(code).not.toBe(0);
    expect(err).toContain("different issuer");
    expect(err).not.toContain("does not offer device login");
  });
});

// The server picks the approval URL and the CLI prints and opens it, so a Galaxy behind a
// misconfigured proxy must not be able to point people at another host (login.ts). The origin must
// match exactly: host, port and scheme, and plain http only to loopback. 76 is EX_PROTOCOL.
describe("the approval page must be on the Galaxy", () => {
  it.each([
    ["another host", "https://evil.example/login/device"],
    ["plain http on the same host", "http://dev.galaxy.example.org/login/device"],
    ["another port", "https://dev.galaxy.example.org:8443/login/device"],
    ["a non-http scheme", "javascript:alert(1)"],
    ["not a URL", "nonsense"],
  ])("refuses %s: nothing shown, opened or polled", async (_n, uri) => {
    const h = harness({ interactive: true });
    h.server.pending.verificationUri = uri;
    const waited = vi.spyOn(h.server.pending, "waitForApproval");
    expect(await loginCommand({ host: HOST_ARG }, h.deps)).toBe(76);
    expect(h.cap.err).toContain("not an https page on dev.galaxy.example.org");
    expect(h.cap.err).toContain("--with-token");
    expect(h.cap.err).not.toContain("One-time code");
    expect(h.opened).toEqual([]);
    expect(waited).not.toHaveBeenCalled();
    expect(h.store.putCalls).toEqual([]);
  });

  // The WHATWG parser drops a tab or newline, so the check passes, but a launcher or a terminal
  // would see them: what is printed and opened is the parser's form.
  it("prints and opens the normalised URL, without a tab or newline the parser strips", async () => {
    const h = harness({ interactive: true });
    h.server.pending.verificationUri = "https://dev.galaxy.example.org/login/de\tvi\nce";
    h.server.pending.outcome = "hang";
    const ac = new AbortController();
    const p = loginCommand({ host: HOST_ARG, signal: ac.signal }, h.deps);
    await vi.waitFor(() => {
      expect(h.cap.err).toContain("Waiting for approval");
    });
    expect(h.cap.err).toContain("Enter it at https://dev.galaxy.example.org/login/device\n");
    h.stdin.write("\n");
    await vi.waitFor(() => {
      expect(h.opened).toEqual(["https://dev.galaxy.example.org/login/device"]);
    });
    ac.abort();
    await p;
  });

  it("allows http to a loopback Galaxy, on its own origin", async () => {
    const h = harness();
    h.server.pending.verificationUri = "http://127.0.0.1:8080/login/device";
    expect(await loginCommand({ host: "http://127.0.0.1:8080" }, h.deps)).toBe(0);
    expect(h.cap.err).toContain("http://127.0.0.1:8080/login/device");
  });
});

// The caller hands `loginCommand` an `AbortSignal` (commands do not touch process signals), and
// every step either takes it or is raced against it. Each test cancels at a different step, since
// each is a different place to be left holding something: a code on the server, a token, a terminal
// in raw mode. 130 is 128 plus SIGINT's 2, the shell's code for a Ctrl-C. A real SIGINT is in
// e2e.int.test.ts.
describe("cancelling with Ctrl-C", () => {
  const expectCancelled = (h: ReturnType<typeof harness>, code: number) => {
    expect(code).toBe(130);
    expect(h.cap.err).toContain("Cancelled");
    expect(h.store.putCalls).toEqual([]);
    expect(h.cap.out).toBe("");
  };

  it("while waiting for approval", async () => {
    const h = harness();
    h.server.pending.outcome = "hang";
    const ac = new AbortController();
    const p = loginCommand({ host: HOST_ARG, signal: ac.signal }, h.deps);
    await vi.waitFor(() => {
      expect(h.cap.err).toContain("Waiting for approval");
    });
    ac.abort();
    expectCancelled(h, await p);
  });

  it("does nothing when the signal is already aborted", async () => {
    const h = harness();
    const ac = new AbortController();
    ac.abort();
    expectCancelled(h, await loginCommand({ host: HOST_ARG, signal: ac.signal }, h.deps));
    expect(h.server.startedWith).toEqual([]);
  });

  it("during discovery, which is told to stop, so no device authorization starts", async () => {
    const h = harness();
    const ac = new AbortController();
    let seen: AbortSignal | undefined;
    h.deps.discover = (_b, signal) => ((seen = signal), never());
    const p = loginCommand({ host: HOST_ARG, signal: ac.signal }, h.deps);
    await vi.waitFor(() => {
      expect(seen).toBeDefined();
    });
    ac.abort();
    expectCancelled(h, await p);
    expect(seen!.aborted).toBe(true);
    expect(h.server.startedWith).toEqual([]);
  });

  it("creates no server-side code when the cancel lands as discovery finishes", async () => {
    const h = harness();
    const ac = new AbortController();
    h.deps.discover = async () => {
      ac.abort();
      return h.server;
    };
    expectCancelled(h, await loginCommand({ host: HOST_ARG, signal: ac.signal }, h.deps));
    expect(h.server.startedWith).toEqual([]);
  });

  it("while the device authorization request is out", async () => {
    const h = harness();
    const ac = new AbortController();
    h.server.startDeviceLogin = (scopes) => (h.server.startedWith.push(scopes), never());
    const p = loginCommand({ host: HOST_ARG, signal: ac.signal }, h.deps);
    await vi.waitFor(() => {
      expect(h.server.startedWith).toHaveLength(1);
    });
    ac.abort();
    expectCancelled(h, await p);
  });

  it("during whoami after approval, which is told to stop, and stores nothing", async () => {
    const h = harness();
    const ac = new AbortController();
    let seen: AbortSignal | undefined;
    h.deps.whoami = (_b, _t, signal) => ((seen = signal), never());
    const p = loginCommand({ host: HOST_ARG, signal: ac.signal }, h.deps);
    await vi.waitFor(() => {
      expect(seen).toBeDefined();
    });
    ac.abort();
    expectCancelled(h, await p);
    expect(seen!.aborted).toBe(true);
  });

  it("without writing the store when the cancel lands as whoami answers", async () => {
    const h = harness();
    const ac = new AbortController();
    h.deps.whoami = async () => {
      ac.abort();
      return { username: "aioue" };
    };
    expectCancelled(h, await loginCommand({ host: HOST_ARG, signal: ac.signal }, h.deps));
  });

  // Once the server has issued a token, a Ctrl-C must still take it back: the revocation is a call
  // of its own, not the login's signal, which has aborted by then. (The in-process server refuses
  // a revoke handed an aborted signal, as the real one does.)
  describe("after the server has issued a token", () => {
    const tmp = useTempDir("gxy-lg-");
    /** Cancel while whoami is out, then return the result of the command. */
    async function cancelDuringWhoami(h: ReturnType<typeof harness>) {
      const ac = new AbortController();
      h.deps.whoami = () => (ac.abort(), never());
      return loginCommand({ host: HOST_ARG, signal: ac.signal }, h.deps);
    }

    it("revokes it, and says it was cancelled and the token revoked", async () => {
      const h = harness();
      expectCancelled(h, await cancelDuringWhoami(h));
      expect(h.server.revoked).toEqual([FAKE_TOKEN]);
      expect(h.cap.err).toContain(
        "Cancelled; the token dev.galaxy.example.org had issued was revoked.",
      );
      expect(h.cap.err).not.toMatch(/could not be kept/);
      expect(h.cap.err).not.toContain("Cancelled.\n"); // said once
      expect(h.cap.err).not.toContain(FAKE_TOKEN);
    });

    it("revokes it when the cancel lands just as whoami answers", async () => {
      const h = harness();
      const ac = new AbortController();
      h.deps.whoami = async () => (ac.abort(), { username: "aioue" });
      expectCancelled(h, await loginCommand({ host: HOST_ARG, signal: ac.signal }, h.deps));
      expect(h.server.revoked).toEqual([FAKE_TOKEN]);
    });

    it("revokes it when Ctrl-C lands during the keyring write, and stores nothing", async () => {
      const h = harness();
      const kr = new FakeKeyring();
      const { file, store } = realStore(tmp.path, kr);
      h.deps.store = store;
      const ac = new AbortController();
      kr.set = async () => {
        ac.abort();
        throw new Error("helper killed by SIGINT");
      };
      expectCancelled(h, await loginCommand({ host: HOST_ARG, signal: ac.signal }, h.deps));
      expect(h.server.revoked).toEqual([FAKE_TOKEN]);
      expect(h.cap.err).not.toContain("Logged in");
      expect(h.cap.err).not.toContain("plain text");
      await expect(readFile(file.path, "utf8")).rejects.toMatchObject({ code: "ENOENT" });
    });

    it("says to revoke it by hand when that fails", async () => {
      const h = harness();
      h.server.revokeError = new Error("no route");
      expectCancelled(h, await cancelDuringWhoami(h));
      expect(h.cap.err).toContain("Cancelled, but dev.galaxy.example.org had issued a token");
      expect(h.cap.err).toContain("revoking it failed");
      expect(h.cap.err).toContain("user preferences");
    });

    it("says to revoke it by hand when the server cannot revoke", async () => {
      const h = harness();
      h.server.supportsRevocation = false;
      expectCancelled(h, await cancelDuringWhoami(h));
      expect(h.server.revoked).toEqual([]);
      expect(h.cap.err).toContain("cannot be revoked from here");
    });

    it("gives up on a revocation that never answers after 5 s", async () => {
      vi.useFakeTimers();
      try {
        const h = harness();
        h.server.revoke = () => never();
        const p = cancelDuringWhoami(h);
        await vi.advanceTimersByTimeAsync(5100);
        expectCancelled(h, await p);
        expect(h.cap.err).toContain("revoking it failed");
      } finally {
        vi.useRealTimers();
      }
    });
  });

  // The same, against the real client and a real Galaxy: the fake `AuthServer` above only proves
  // what login asks for, and the signal that broke this was baked into the real one by `discover`.
  describe("against a real Galaxy", () => {
    const galaxyOf = useFakeGalaxies();

    it("takes the token back when Ctrl-C lands during whoami, so it no longer works", async () => {
      const galaxy = await galaxyOf();
      const h = harness();
      const ac = new AbortController();
      h.deps.discover = (u: BaseUrl, signal?: AbortSignal) =>
        discover(u, { signal, minIntervalMs: 0 });
      h.deps.whoami = () => (ac.abort(), never());
      const p = loginCommand({ host: galaxy.baseUrl, signal: ac.signal }, h.deps);
      await vi.waitFor(() => {
        expect(galaxy.pendingCodes()).toHaveLength(1);
      });
      galaxy.approve(galaxy.pendingCodes()[0]!);
      expectCancelled(h, await p);
      const [token] = galaxy.issuedTokens();
      expect(token).toBeDefined();
      expect(galaxy.isTokenValid(token!)).toBe(false);
      expect(h.cap.err).toContain("Cancelled; the token");
    });
  });

  it("at the Galaxy URL prompt", async () => {
    const h = harness({ interactive: true });
    const ac = new AbortController();
    const p = loginCommand({ signal: ac.signal }, h.deps);
    await vi.waitFor(() => {
      expect(h.cap.err).toContain("Galaxy URL");
    });
    ac.abort();
    expectCancelled(h, await p);
  });

  // The keyring write is the one step that can last (an unlock prompt), and Ctrl-C kills its helper.
  // That must cancel, not fall back to a plaintext token in hosts.yml with "Logged in".
  describe("during the keyring write", () => {
    const tmp = useTempDir("gxy-lg-");

    it("exits 130, says Cancelled and stores nothing", async () => {
      const h = harness({ stdinText: `${FAKE_TOKEN}\n` });
      const kr = new FakeKeyring();
      const { file, store } = realStore(tmp.path, kr);
      h.deps.store = store;
      const ac = new AbortController();
      kr.set = async () => {
        ac.abort();
        throw new Error("helper killed by SIGINT");
      };
      expect(
        await loginCommand({ host: HOST_ARG, withToken: true, signal: ac.signal }, h.deps),
      ).toBe(130);
      expect(h.cap.err).toContain("Cancelled.");
      expect(h.cap.err).not.toContain("Logged in");
      expect(h.cap.err).not.toContain("plain text");
      expect(h.cap.err + h.cap.out).not.toContain(FAKE_TOKEN);
      expect(h.cap.out).toBe("");
      await expect(readFile(file.path, "utf8")).rejects.toMatchObject({ code: "ENOENT" });
      expect(kr.data.size).toBe(0);
    });
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
  it("reads the key from stdin, trims it, and stores a pasted login without using the device flow", async () => {
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
    expect(h.server.startedWith).toEqual([]);
    expect(h.cap.err).toContain("Logged in to dev.galaxy.example.org as aioue");
    expect(h.cap.out).toBe("");
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

// An admin token is never written to the plaintext file unless `--insecure-storage` asks for it
// (login.ts), since a stolen one carries full admin rights. With no keyring it is refused with 77,
// after the warning, rather than quietly saved to a file.
describe("tokens that carry admin rights", () => {
  const admin = { username: "root", isAdmin: true };

  it("get a warning, printed once, and the login is still stored", async () => {
    const h = harness();
    h.knobs.whoami = admin;
    expect(await loginCommand({ host: HOST_ARG }, h.deps)).toBe(0);
    expect(h.cap.err.match(/full admin rights on dev\.galaxy\.example\.org/g)).toHaveLength(1);
    expect(h.store.logins.size).toBe(1);
    // The admin put is what makes the store refuse a plaintext fallback.
    expect(h.store.putCalls[0]!.opts).toMatchObject({ requireKeyring: true });
  });

  it("do not make an ordinary user's login warn", async () => {
    const h = harness();
    await loginCommand({ host: HOST_ARG }, h.deps);
    expect(h.cap.err).not.toContain("admin rights");
    expect(h.store.putCalls[0]!.opts?.requireKeyring).toBeUndefined();
  });

  it.each([
    ["device", false],
    ["pasted", true],
  ])(
    "(%s) are refused with 77, after the warning, when there is no keyring",
    async (_how, withToken) => {
      const h = harness({ stdinText: `${FAKE_TOKEN}\n` });
      h.store.keyringAvailable = false;
      h.knobs.whoami = admin;
      expect(await loginCommand({ host: HOST_ARG, withToken }, h.deps)).toBe(77);
      expect(h.store.logins.size).toBe(0);
      const warned = h.cap.err.indexOf("full admin rights");
      expect(warned).toBeGreaterThanOrEqual(0);
      expect(h.cap.err.indexOf(REFUSED_WITHOUT_KEYRING)).toBeGreaterThan(warned);
      expect(h.cap.err).not.toContain(FAKE_TOKEN);
    },
  );

  it("are kept in the file when --insecure-storage says so", async () => {
    const h = harness();
    h.store.keyringAvailable = false;
    h.knobs.whoami = admin;
    expect(await loginCommand({ host: HOST_ARG, insecureStorage: true }, h.deps)).toBe(0);
    expect(h.store.logins.get(HOST)?.storage).toBe("file");
    expect(h.store.putCalls[0]!.opts).toMatchObject({
      insecureStorage: true,
      requireKeyring: true,
    });
  });

  it("do not stop an ordinary user falling back to the file", async () => {
    const h = harness();
    h.store.keyringAvailable = false;
    expect(await loginCommand({ host: HOST_ARG }, h.deps)).toBe(0);
    expect(h.store.logins.size).toBe(1);
  });

  it("add their warning to a pasted key's own no-expiry warning", async () => {
    const plain = harness({ stdinText: `${FAKE_TOKEN}\n` });
    await loginCommand({ host: HOST_ARG, withToken: true }, plain.deps);
    expect(plain.cap.err).toContain("no expiry");
    expect(plain.cap.err).not.toContain("admin rights");

    const root = harness({ stdinText: `${FAKE_TOKEN}\n` });
    root.knobs.whoami = admin;
    await loginCommand({ host: HOST_ARG, withToken: true }, root.deps);
    expect(root.cap.err).toContain("admin rights");
    expect(root.cap.err).toContain("no expiry");
  });
});
