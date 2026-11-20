// `createCredentialStore`, the one factory, with the keyring child process replaced by an
// in-memory fake that speaks keyring-child.ts's stdin/stdout protocol. This is what stops any test
// here reaching a real keyring, and it counts spawns so "never touched" can be asserted.
import { beforeEach, describe, expect, it, vi } from "vitest";

/**
 * What the fake child process records: how many were started, and the secrets it holds, keyed
 * `service|account`. Hoisted so that the `vi.mock` factory below, which runs before the imports,
 * can use it.
 */
const state = vi.hoisted(() => ({ spawned: 0, secrets: new Map<string, string>() }));
// A fake `spawn`. Each call counts a start and returns a child that reads the JSON request from
// stdin until it ends, applies it to `state.secrets` (`get` answers the stored value or null, `set`
// stores, `delete` removes) and replies `{ ok: true, value }` on stdout before it closes. This is
// the protocol `keyring-child.ts` speaks, so the real `KeyringStore` and `nativeBackend` run
// unchanged on top of it.
vi.mock("node:child_process", async () => {
  const { EventEmitter } = await import("node:events");
  const { PassThrough } = await import("node:stream");
  return {
    spawn: () => {
      state.spawned++;
      const child = Object.assign(new EventEmitter(), {
        stdin: new PassThrough(),
        stdout: new PassThrough(),
        kill: () => {},
      });
      let body = "";
      child.stdin.on("data", (d: Buffer) => (body += d.toString()));
      child.stdin.on("end", () => {
        const r = JSON.parse(body) as {
          op: string;
          service: string;
          account: string;
          secret?: string;
        };
        const k = `${r.service}|${r.account}`;
        const value =
          r.op === "get"
            ? (state.secrets.get(k) ?? null)
            : r.op === "set"
              ? (state.secrets.set(k, r.secret!), true)
              : state.secrets.delete(k);
        child.stdout.end(JSON.stringify({ ok: true, value }));
        setImmediate(() => child.emit("close", 0));
      });
      return child;
    },
  };
});

const { createCredentialStore } = await import("../../src/auth/store");
const { loadSourcesAsync, resolveConnection } = await import("../../src/config");
const { DAY, FAKE_TOKEN, HOST, fakeLogin, useTempDir } = await import("./harness");

const tmp = useTempDir("gxy-nk-");
// Reset for every test, so a count or a secret from one cannot satisfy another's assertion.
beforeEach(() => {
  state.spawned = 0;
  state.secrets.clear();
});
/** An environment for `createCredentialStore` that keeps the config in this test's temp directory. */
const env = (extra: Record<string, string> = {}) => ({
  GXY_CONFIG_DIR: `${tmp.path}/cfg`,
  ...extra,
});

// The variable turns the keyring off for any value except empty, "0" and "false" (in any case).
// The accepted and refused values are written out here, not read from the code. A keyring that is
// off must never start a child, so the spawn count is what is checked.
describe("GXY_NO_KEYRING", () => {
  it.each(["1", "yes", "true"])(
    "=%s keeps tokens in the file, with a warning, and never starts the keyring",
    async (v) => {
      const s = createCredentialStore(env({ GXY_NO_KEYRING: v }), tmp.path, () => {});
      const res = await s.put(fakeLogin());
      expect(res.storage).toBe("file");
      expect(res.warning).toMatch(/GXY_NO_KEYRING is set/);
      expect((await s.get(HOST))?.token).toBe(FAKE_TOKEN);
      expect(state.spawned).toBe(0);
    },
  );

  it.each(["", "0", "false", "False"])("=%j leaves the keyring on", async (v) => {
    const s = createCredentialStore(env({ GXY_NO_KEYRING: v }), tmp.path, () => {});
    expect((await s.put(fakeLogin())).storage).toBe("keyring");
    expect(state.spawned).toBeGreaterThan(0);
    expect(state.secrets.size).toBe(1);
  });

  it("unset leaves the keyring on", async () => {
    expect((await createCredentialStore(env(), tmp.path, () => {}).put(fakeLogin())).storage).toBe(
      "keyring",
    );
  });
});

describe("the native keyring is touched only when a record needs it", () => {
  // A file-stored login never needs the keyring, and starting a child on a machine with a wedged bus
  // would cost a timeout per call for nothing. The token is also read through `loadSourcesAsync`
  // and `resolveConnection`, the lookup an operation makes, so the check covers operations and not
  // only the `auth` commands.
  // `now` is fixed to the fixture's own clock: the login's `expiresAt` is a fixed date, and an expired
  // login is not used. `base` is empty so no `.env` or profile of the machine's can supply a key.
  it("a file-stored login never starts a keyring child (put, list, get, an operation's lookup, delete)", async () => {
    const store = createCredentialStore(env(), tmp.path, () => {});
    const login = fakeLogin();
    expect((await store.put(login, { insecureStorage: true })).storage).toBe("file");
    expect(await store.list()).toHaveLength(1);
    expect((await store.get(login.baseUrl))?.token).toBe(login.token);
    const src = await loadSourcesAsync(
      {},
      { store, now: new Date(DAY), base: { env: {}, dotenv: {}, profiles: {} } },
    );
    expect(resolveConnection({}, src).apiKey).toBe(login.token);
    expect(await store.delete(login.baseUrl)).toBe(true);
    expect(state.spawned).toBe(0);
  });

  it("a keyring login round-trips through the native backend", async () => {
    const store = createCredentialStore(env(), tmp.path, () => {});
    expect((await store.put(fakeLogin())).storage).toBe("keyring");
    expect((await store.get(HOST))?.token).toBe(FAKE_TOKEN);
    expect(await store.delete(HOST)).toBe(true);
    expect(state.secrets.size).toBe(0);
  });
});
