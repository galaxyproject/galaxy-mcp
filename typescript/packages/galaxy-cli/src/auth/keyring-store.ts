import { spawn, type ChildProcess } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import type { BaseUrl } from "./types";

/**
 * The OS keyring, behind a seam tests can fake. The real backend is `@napi-rs/keyring`
 * pinned to the Secret Service (gnome-keyring, KWallet, ...) and never the kernel keyutils
 * keyring.
 *
 * Why pinned: left to itself the binding tries Secret Service, then silently falls back to
 * keyutils on any error (keyring-node#142). Observed on Linux in a container with no D-Bus
 * session: the default `Entry` constructor throws `PermissionDenied` (the container blocks the
 * keyctl syscalls), while `{ linux: { store: "secret-service" } }` throws a D-Bus error and
 * `"keyutils"` the same `PermissionDenied`. On a host where keyutils does work, the binding's
 * own README says it is in-memory only; it lives in the kernel session/user keyring and is gone
 * at reboot (and, for a session keyring, when the login session ends). A store that appears to
 * work but loses tokens between shells is worse than the plaintext file fallback, so we refuse
 * it: when there is no Secret Service, the keyring is "unavailable" and `store.ts` uses the file.
 *
 * Availability is detected by use, not by a separate probe: `store.ts` writes the token and
 * reads it back, and falls back to the file if either fails. A probe would only say the keyring
 * worked a moment ago. Every call has a timeout, because a locked or wedged Secret Service can
 * wait forever on a prompt nobody sees.
 *
 * The native module runs in a child process per call (`keyring-child.ts`) that is SIGKILLed on
 * the timeout: its constructor can block forever on a dead D-Bus, which nothing in-process can
 * interrupt. So a broken bus costs at most one timeout per command (a store that has seen one
 * timeout fails the rest of its calls at once), and never keeps the CLI alive.
 */
export interface KeyringBackend {
  get(service: string, account: string, opts?: CallOptions): Promise<string | null>;
  set(service: string, account: string, secret: string, opts?: CallOptions): Promise<void>;
  /** Resolves false when there was nothing to delete. */
  delete(service: string, account: string, opts?: CallOptions): Promise<boolean>;
  /**
   * True when the backend enforces the timeout it is given by itself and kills what it started
   * (the native one). `KeyringStore` then does not race its own timer against it: that timer
   * would fire a moment before the child is killed, and a caller that exits on the rejection
   * would leave the child blocked on the bus. A backend without it (a fake) is raced.
   */
  readonly selfTimed?: boolean;
}

interface CallOptions {
  /** Give up after this long, instead of the backend's default. */
  timeoutMs?: number;
}

/**
 * How long one keyring call may take before it is given up on when nobody can answer a prompt
 * (a script, a pipe, an agent). A child starts and loads the module in about 20 ms, so this is
 * only ever spent on a keyring that is wedged or waiting on a prompt.
 */
const KEYRING_TIMEOUT_MS = 5000;

/**
 * The same, with a person at the terminal. A locked Secret Service (a GNOME login keyring after
 * an SSH session, KWallet's first-use wizard) asks for a password and blocks the call until it
 * is typed, and 5 s would kill the prompt and drop the login into the plaintext file. gh allows
 * 60 s. The cost is that a keyring that is really dead takes this long to be given up on, once
 * per command, which is why the person is told while they wait (`notify`).
 */
const KEYRING_INTERACTIVE_TIMEOUT_MS = 60_000;

/** How long a call may run before the person is told that it is waiting on the keyring. */
const NOTICE_AFTER_MS = 1500;

/** How much of a child's reply is kept: a real one is under a hundred bytes. */
const MAX_REPLY_BYTES = 64 * 1024;

/**
 * Any reason the keyring could not be used for a call: no Secret Service, a call that went
 * wrong, a timeout, a delete that did not take. `store.ts` treats all of them as "no usable
 * keyring" and falls back to the file (or, for an admin token, refuses), never as a crash.
 */
export class KeyringUnavailableError extends Error {
  /** The call ran out of time, as opposed to being refused: a keyring that stays silent is wedged. */
  readonly timedOut: boolean;
  constructor(message: string, options?: { cause?: unknown; timedOut?: boolean }) {
    super(message, options);
    this.name = "KeyringUnavailableError";
    this.timedOut = options?.timedOut ?? false;
  }
}

/**
 * A keyring that is never there, for `GXY_NO_KEYRING=1`: tokens go to the file, as on a headless
 * box. The test suite sets it so that no test can reach the real keyring of whoever runs it.
 */
export const noKeyringBackend: KeyringBackend = {
  get: () => Promise.reject(new KeyringUnavailableError("disabled by GXY_NO_KEYRING")),
  set: () => Promise.reject(new KeyringUnavailableError("disabled by GXY_NO_KEYRING")),
  delete: () => Promise.reject(new KeyringUnavailableError("disabled by GXY_NO_KEYRING")),
};

/**
 * `gxy:<baseUrl>`, with the username as the account: one entry per login, the layout gh uses
 * (`gh:<hostname>`). The whole normalised URL rather than the host, so two path prefixes on one
 * host are two entries, as they are two keys in `hosts.yml`.
 */
function keyringService(baseUrl: BaseUrl): string {
  return `gxy:${baseUrl}`;
}

interface KeyringStoreOptions {
  /**
   * A person is at the terminal and may be asked to unlock the keyring, so calls get
   * `interactiveTimeoutMs` and the person is told when one is slow. Passed in by the caller,
   * which knows its terminal; this class never looks at one.
   */
  interactive?: boolean;
  interactiveTimeoutMs?: number;
  /** Where the "waiting for the keyring" notice goes (stderr); none means no notice. */
  notify?: (message: string) => void;
  noticeAfterMs?: number;
}

/**
 * One store lives for one command. After a call has timed out the rest fail at once: a keyring
 * that did not answer is not going to answer the next call either, and each further wait is
 * another timeout on a login (a failed write, then its cleanup) or a logout (a delete and its
 * check).
 */
export class KeyringStore {
  private readonly backend: KeyringBackend;
  private readonly timeoutMs: number;
  private readonly opts: KeyringStoreOptions;
  private wedged = false;
  private noticed = false;

  // No parameter properties: the integration test runs this file as source under Node's type
  // stripping, which does not support them.
  constructor(
    backend?: KeyringBackend,
    timeoutMs: number = KEYRING_TIMEOUT_MS,
    opts: KeyringStoreOptions = {},
  ) {
    this.timeoutMs = timeoutMs;
    this.opts = opts;
    this.backend = backend ?? nativeBackend({ timeoutMs });
  }

  /** True for `GXY_NO_KEYRING`: the keyring is off by request, which is not the same as broken. */
  get disabled(): boolean {
    return this.backend === noKeyringBackend;
  }

  get(baseUrl: BaseUrl, account: string): Promise<string | null> {
    return this.call("read", (o) => this.backend.get(keyringService(baseUrl), account, o));
  }

  set(baseUrl: BaseUrl, account: string, secret: string): Promise<void> {
    return this.call("write", (o) => this.backend.set(keyringService(baseUrl), account, secret, o));
  }

  /**
   * Delete, then read back and require nothing there. Some backends report success for a
   * delete that did not happen (a locked collection, a different item matched); the binding
   * itself once folded native errors into `false`, which reads as "nothing to delete"
   * (keyring-node#137). A token that quietly stays in the keyring after `logout` is the worst
   * outcome, so "deleted" has to be observed, not assumed.
   */
  async delete(baseUrl: BaseUrl, account: string): Promise<boolean> {
    const service = keyringService(baseUrl);
    const existed = await this.call("delete", (o) => this.backend.delete(service, account, o));
    const remaining = await this.call("check after delete", (o) =>
      this.backend.get(service, account, o),
    );
    if (remaining !== null) {
      throw new KeyringUnavailableError(
        "The system keyring still holds the secret after a delete, so it may remain there.",
      );
    }
    return existed;
  }

  /**
   * Run a backend call with its timeout, so that no backend can hang a caller. The native
   * backend enforces the timeout itself and kills its child; any other (a fake) is raced here.
   * While a person waits, a notice says what for, but only once a call has been slow for a
   * moment: a keyring that answers in 20 ms has nothing to announce, and the notice on every
   * login would be noise.
   */
  private async call<T>(what: string, fn: (o: CallOptions) => Promise<T>): Promise<T> {
    if (this.wedged) {
      throw new KeyringUnavailableError(
        `The system keyring did not answer earlier, so a ${what} was not tried.`,
        { timedOut: true },
      );
    }
    const timeoutMs = this.opts.interactive
      ? (this.opts.interactiveTimeoutMs ?? KEYRING_INTERACTIVE_TIMEOUT_MS)
      : this.timeoutMs;
    let timer: NodeJS.Timeout | undefined;
    let noticeTimer: NodeJS.Timeout | undefined;
    if (this.opts.interactive && this.opts.notify && !this.noticed) {
      const notify = this.opts.notify;
      noticeTimer = setTimeout(() => {
        this.noticed = true;
        notify("Waiting for the system keyring (unlock it if asked)...");
      }, this.opts.noticeAfterMs ?? NOTICE_AFTER_MS);
    }
    try {
      const run = fn({ timeoutMs });
      if (this.backend.selfTimed) return await run;
      const timeout = new Promise<never>((_, reject) => {
        timer = setTimeout(() => {
          reject(
            new KeyringUnavailableError(
              `The system keyring did not answer a ${what} within ${timeoutMs} ms.`,
              { timedOut: true },
            ),
          );
        }, timeoutMs);
      });
      return await Promise.race([run, timeout]);
    } catch (e) {
      if (e instanceof KeyringUnavailableError) {
        if (e.timedOut) this.wedged = true;
        throw e;
      }
      throw new KeyringUnavailableError(
        `The system keyring failed a ${what}: ${e instanceof Error ? e.message : String(e)}`,
        { cause: e },
      );
    } finally {
      clearTimeout(timer);
      clearTimeout(noticeTimer);
    }
  }
}

interface NativeBackendOptions {
  /** Kill the child and give up after this long. */
  timeoutMs?: number;
  /** The child script; found next to this module by default. Tests point it at a stub. */
  script?: string;
  /** Environment for the child; a minimal one built from `process.env` by default. */
  env?: NodeJS.ProcessEnv;
}

/** The child lives beside this module: `dist/keyring-child.js` once built, `keyring-child.ts` from source. */
function childScript(): string {
  const here = import.meta.url;
  for (const name of ["keyring-child.js", "keyring-child.ts"]) {
    const p = fileURLToPath(new URL(name, here));
    if (existsSync(p)) return p;
  }
  return fileURLToPath(new URL("keyring-child.js", here));
}

/**
 * What the child needs to reach the session bus and the OS, and nothing else: no API keys or
 * tokens, so a `GALAXY_API_KEY` in the CLI's environment never reaches a process that has no
 * use for it.
 */
const CHILD_ENV_KEYS = [
  "HOME",
  "USER",
  "LOGNAME",
  "PATH",
  "TMPDIR",
  "LANG",
  "LC_ALL",
  "DBUS_SESSION_BUS_ADDRESS",
  "XDG_RUNTIME_DIR",
  "XDG_DATA_DIRS",
  "XDG_CURRENT_DESKTOP",
  "XDG_SESSION_ID",
  "DISPLAY",
  "WAYLAND_DISPLAY",
  // Windows locates the credential store through these.
  "USERPROFILE",
  "APPDATA",
  "LOCALAPPDATA",
  "SystemRoot",
  "SYSTEMROOT",
];

export function childEnv(source: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv {
  const out: NodeJS.ProcessEnv = {};
  for (const k of CHILD_ENV_KEYS) if (source[k] !== undefined) out[k] = source[k];
  return out;
}

type ChildReply =
  | { ok: true; value: string | boolean | null }
  | { ok: false; kind: "unavailable" | "not-found" | "failed"; message: string };

/**
 * A reply that is complete and has the shape this operation can produce, else undefined. Only the
 * helper writes these, but the delete re-check relies on them, so a reply of the wrong shape is
 * "no usable answer", never read as "nothing there": `get` answers a string or null, `set`
 * `true`, `delete` a boolean, and a refusal names a known kind and carries a message.
 */
function parseReply(op: "get" | "set" | "delete", text: string): ChildReply | undefined {
  let r: unknown;
  try {
    r = JSON.parse(text);
  } catch {
    return undefined;
  }
  if (typeof r !== "object" || r === null) return undefined;
  const o = r as Record<string, unknown>;
  if (o.ok === true) {
    const v = o.value;
    if (op === "get" && (typeof v === "string" || v === null)) return { ok: true, value: v };
    if (op === "set" && (v === true || v === undefined)) return { ok: true, value: true };
    if (op === "delete" && typeof v === "boolean") return { ok: true, value: v };
    return undefined;
  }
  if (o.ok === false) {
    const kind = o.kind;
    if (
      (kind === "unavailable" || kind === "not-found" || kind === "failed") &&
      typeof o.message === "string"
    ) {
      return { ok: false, kind, message: o.message };
    }
  }
  return undefined;
}

/**
 * Helpers still running. A helper blocked on a dead bus never exits by itself, and only this
 * process's timeout kills it, so a gxy that is ended another way must take it down first:
 * `timeout`, a CI cancel or an agent sends SIGTERM (or SIGINT, SIGHUP) to gxy alone, not to its
 * process group as Ctrl-C at a terminal does. SIGKILL can't be caught, so a `kill -9` of gxy can
 * still leave one behind.
 */
const liveHelpers = new Set<ChildProcess>();
const REAPED_SIGNALS = ["SIGINT", "SIGTERM", "SIGHUP"] as const;

const killLiveHelpers = (): void => {
  for (const c of liveHelpers) c.kill("SIGKILL");
  liveHelpers.clear();
};

// Runs first (prepended), and steps aside before the other listeners run, so the signal then does
// what it would have done without it: a command's own Ctrl-C handler cancels, signal-exit (which a
// dependency loads, and which ends the process only when its listeners are the only ones) ends it,
// and with no listener left the signal is raised again to end the process.
const onReapedSignal = (sig: NodeJS.Signals): void => {
  killLiveHelpers();
  unhookHelpers();
  if (process.listenerCount(sig) === 0) process.kill(process.pid, sig);
};

function hookHelpers(): void {
  process.on("exit", killLiveHelpers);
  for (const sig of REAPED_SIGNALS) process.prependListener(sig, onReapedSignal);
}

function unhookHelpers(): void {
  process.removeListener("exit", killLiveHelpers);
  for (const sig of REAPED_SIGNALS) process.removeListener(sig, onReapedSignal);
}

function trackHelper(child: ChildProcess): void {
  if (liveHelpers.size === 0) hookHelpers();
  liveHelpers.add(child);
  const untrack = () => {
    if (liveHelpers.delete(child) && liveHelpers.size === 0) unhookHelpers();
  };
  child.once("close", untrack);
  child.once("error", untrack);
}

/**
 * The real backend: one child process per call, running the native keyring module. Only reached
 * when a record says `keyring` or login probes it, so file-stored logins never spawn it.
 * Starting the child and loading the module costs about 20 ms a call; a call that outlives the
 * timeout is SIGKILLed.
 *
 * A call is settled as soon as a complete reply has arrived, and the child is killed then: a
 * helper that has answered but cannot exit (a native thread still blocked on the bus) must not
 * turn a good answer into a timeout, which for a `set` would also make a stored token look like
 * a failed write.
 */
export function nativeBackend(opts: NativeBackendOptions = {}): KeyringBackend {
  const defaultTimeoutMs = opts.timeoutMs ?? KEYRING_TIMEOUT_MS;
  const call = (
    op: "get" | "set" | "delete",
    service: string,
    account: string,
    secret: string | undefined,
    timeoutMs: number,
  ) =>
    new Promise<ChildReply>((resolve, reject) => {
      // Not shell, not argv: argv is readable by every user in `ps` and `/proc`, so the secret
      // goes down stdin only.
      const child = spawn(process.execPath, [opts.script ?? childScript()], {
        env: opts.env ?? childEnv(),
        stdio: ["pipe", "pipe", "ignore"],
        windowsHide: true,
      });
      trackHelper(child);
      let out = "";
      // Settle once: the kill on timeout also makes `close` fire, and must not turn that
      // timeout into "no usable answer".
      let settled = false;
      const finish = (fn: () => void) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        fn();
      };
      const noAnswer = () => {
        reject(new KeyringUnavailableError(`The keyring helper gave no usable answer to a ${op}.`));
      };
      const timer = setTimeout(() => {
        child.kill("SIGKILL");
        finish(() => {
          reject(
            new KeyringUnavailableError(
              `The system keyring did not answer a ${op} within ${timeoutMs} ms.`,
              { timedOut: true },
            ),
          );
        });
      }, timeoutMs);
      child.stdout.setEncoding("utf8");
      child.stdout.on("data", (d: string) => {
        if (settled) return;
        out += d;
        if (out.length > MAX_REPLY_BYTES) {
          child.kill("SIGKILL");
          finish(noAnswer);
          return;
        }
        const reply = parseReply(op, out);
        if (reply) {
          child.kill("SIGKILL");
          finish(() => {
            resolve(reply);
          });
        }
      });
      child.on("error", (e) => {
        finish(() => {
          reject(
            new KeyringUnavailableError(`The keyring helper could not start: ${e.message}`, {
              cause: e,
            }),
          );
        });
      });
      // Garbage, silence and a crash all end up here, and all mean "unavailable".
      child.on("close", () => {
        finish(() => {
          const reply = parseReply(op, out);
          if (reply) resolve(reply);
          else noAnswer();
        });
      });
      // The child may die before reading; its exit is reported by close, so ignore EPIPE here.
      child.stdin.on("error", () => {});
      child.stdin.end(
        JSON.stringify({ op, service, account, ...(secret === undefined ? {} : { secret }) }),
      );
    });
  const value = async (
    op: "get" | "set" | "delete",
    service: string,
    account: string,
    secret: string | undefined,
    o: CallOptions | undefined,
  ) => {
    const r = await call(op, service, account, secret, o?.timeoutMs ?? defaultTimeoutMs);
    if (r.ok) return r.value;
    if (r.kind === "not-found") return null;
    if (r.kind === "unavailable") throw new KeyringUnavailableError(r.message);
    throw new Error(r.message);
  };
  return {
    selfTimed: true,
    async get(service, account, o) {
      const v = await value("get", service, account, undefined, o);
      return typeof v === "string" ? v : null;
    },
    async set(service, account, secret, o) {
      await value("set", service, account, secret, o);
    },
    async delete(service, account, o) {
      return (await value("delete", service, account, undefined, o)) === true;
    },
  };
}
