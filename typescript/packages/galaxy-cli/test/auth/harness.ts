/**
 * The fakes for running an `auth` command in process. A command is a function over `AuthDeps`, so
 * a test hands it in-memory stand-ins for the store, the server, the clock and the terminal, steers
 * each one, and reads back what the command did to it. What only a real process, pipe or socket can
 * show belongs in `e2e.int.test.ts` and `test/support/fake-galaxy.ts`, not here.
 */
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";
import { afterEach, beforeEach } from "vitest";
import { normalizeBaseUrl } from "../../src/auth/base-url";
import { resolveHost } from "../../src/auth/host";
import {
  AuthError,
  type AuthDeps,
  type BaseUrl,
  type CredentialStore,
  type LoginSummary,
  type StorageKind,
  type StoredLogin,
  type WhoAmI,
} from "../../src/auth/types";

/** The clock every harness runs on, so "expires in 7h" does not depend on when the suite runs. */
export const DAY = "2026-11-06T10:44:00.000Z";
export const HOST: BaseUrl = normalizeBaseUrl("https://dev.galaxy.example.org");
export const OTHER: BaseUrl = normalizeBaseUrl("https://other.example.org");
/** Obviously fake. */
export const FAKE_TOKEN = "fake-token-0123456789abcdef";

/** About eight hours after `DAY`: for tests that need a login with an expiry (`expiresAt: SOON`). */
export const SOON = "2026-11-06T18:42:00.000Z";

/** A stored pasted-key login for `HOST`, with no expiry; `over` replaces any field. */
export function fakeLogin(over: Partial<StoredLogin> = {}): StoredLogin {
  return {
    baseUrl: HOST,
    username: "aioue",
    token: FAKE_TOKEN,
    method: "pasted",
    scopes: [],
    expiresAt: null,
    obtainedAt: DAY,
    ...over,
  };
}

/**
 * A credential store in memory. It answers from `logins` and records every `put`, so a test can
 * show that nothing was stored. What the real store does is tested in store.test.ts, not copied
 * here. Like the real one, `list` leaves the token out.
 */
export class FakeStore implements CredentialStore {
  logins = new Map<string, StoredLogin & { storage: StorageKind }>();
  putCalls: { login: StoredLogin }[] = [];
  /** Set to make `put` throw this instead (a failed write). */
  putError: Error | undefined;

  seed(login: StoredLogin): this {
    this.logins.set(login.baseUrl, { ...login, storage: "file" });
    return this;
  }
  async get(baseUrl: BaseUrl) {
    return this.logins.get(baseUrl) ?? null;
  }
  async put(login: StoredLogin): Promise<void> {
    this.putCalls.push({ login });
    if (this.putError) throw this.putError;
    this.logins.set(login.baseUrl, { ...login, storage: "file" });
  }
  async delete(baseUrl: BaseUrl) {
    return this.logins.delete(baseUrl);
  }
  async list(): Promise<LoginSummary[]> {
    return [...this.logins.values()]
      .sort((a, b) => a.baseUrl.localeCompare(b.baseUrl))
      .map(({ token: _t, ...rest }) => rest);
  }
}

/** A temp directory per test, removed afterwards. Call it at the top of a file or a `describe`. */
export function useTempDir(prefix = "gxy-test-"): { readonly path: string } {
  let dir = "";
  beforeEach(async () => {
    // Under the run's root (`global-setup.ts` removes it at the end), so a killed worker leaves nothing.
    dir = await mkdtemp(join(process.env.GXY_TEST_ROOT ?? tmpdir(), prefix));
  });
  afterEach(async () => {
    await rm(dir, { recursive: true, force: true });
  });
  return {
    get path() {
      return dir;
    },
  };
}

/**
 * What a command wrote, kept as two strings. Commands write through `deps.io` and never to
 * `console`, so nothing needs spying on and each run's output is its own. stdout and stderr stay
 * apart because that split is the contract: stdout is what a script consumes (a token, `--json`)
 * and stderr is everything a person reads, so most tests assert that one of them is empty.
 */
export class Capture {
  out = "";
  err = "";
  stdout = { write: (s: string) => void (this.out += s) };
  stderr = { write: (s: string) => void (this.err += s) };
}

export interface Harness {
  deps: AuthDeps;
  store: FakeStore;
  cap: Capture;
  stdin: PassThrough;
  /** Each live check, with the token it was made with, so a test can show one was skipped. */
  whoamiCalls: { baseUrl: string; token: string }[];
  /** What a test flips after building: the live check's answer (or its `AuthError`). */
  knobs: { whoami: WhoAmI | AuthError };
}

/**
 * A fresh set of in-memory fakes behind `AuthDeps`. `stdinText` is written and ended at once;
 * without it stdin stays open and empty, which is how a command that is waiting on input gets
 * tested, and cancelled. The clock is fixed at `DAY`.
 */
export function harness(
  opts: {
    interactive?: boolean;
    env?: Record<string, string>;
    stdinText?: string;
    stored?: StoredLogin[];
    hostFromEnv?: string;
    ignoredLoginHost?: { source: "dotenv" | "profile"; url: string };
  } = {},
): Harness {
  const store = new FakeStore();
  for (const l of opts.stored ?? []) store.seed(l);
  const now = () => new Date(DAY);
  const cap = new Capture();
  const stdin = new PassThrough();
  if (opts.stdinText !== undefined) {
    stdin.end(opts.stdinText);
  }
  const whoamiCalls: { baseUrl: string; token: string }[] = [];
  const knobs: Harness["knobs"] = { whoami: { username: "aioue", email: "a@example.org" } };
  const env = { ...(opts.hostFromEnv ? { GALAXY_URL: opts.hostFromEnv } : {}), ...opts.env };
  const deps: AuthDeps = {
    store,
    async whoami(baseUrl, token) {
      whoamiCalls.push({ baseUrl, token });
      if (knobs.whoami instanceof AuthError) throw knobs.whoami;
      return knobs.whoami;
    },
    // The real rule, over the fake store, so a host is picked here as operations pick it (host.ts).
    // `.env` and profiles are not in play: `ignoredLoginHost` is a canned answer, and the real
    // read of a working directory's `.env` is shown by e2e.int.test.ts.
    resolveHost: (explicit, purpose) =>
      resolveHost({
        explicit,
        env,
        dotenv: {},
        store,
        forLogin: purpose === "login",
      }),
    ignoredLoginHost: () => opts.ignoredLoginHost,
    now,
    env,
    io: { stdin, stdout: cap.stdout, stderr: cap.stderr, interactive: opts.interactive ?? false },
    commandName: "gxy",
  };
  return { deps, store, cap, stdin, whoamiCalls, knobs };
}

/** The `AuthError` a promise rejects with (the test fails if it resolves or rejects with anything else). */
export async function authErrorOf(p: Promise<unknown>): Promise<AuthError> {
  const e = await p.then(
    () => undefined,
    (x: unknown) => x,
  );
  if (!(e instanceof AuthError))
    throw new Error(
      `expected an AuthError, got ${e === undefined ? "a resolved promise" : e instanceof Error ? e.message : JSON.stringify(e)}`,
    );
  return e;
}
