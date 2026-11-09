import { checkExpiry, describeExpiry } from "../expiry";
import {
  AuthError,
  type AuthDeps,
  type KeySourceId,
  type LoginSummary,
  type StoredLogin,
} from "../types";
import {
  fail,
  formatExpiry,
  formatTimeLeft,
  hostOf,
  maskToken,
  describeStorage,
  sanitize,
} from "./common";
import { abortReason } from "../abort";

export interface StatusOptions {
  host?: string;
  json?: boolean;
  showToken?: boolean;
  /** Aborted on Ctrl-C by the caller (commands don't touch process signals). */
  signal?: AbortSignal;
  /** Per host; a slow Galaxy is that host's failure, not the whole command's. */
  timeoutMs?: number;
}

const CHECK_TIMEOUT_MS = 10_000;
/** 128 + SIGINT, which is what a shell reports for a process that Ctrl-C ended. */
const EX_INTERRUPTED = 130;

/** Settle with `p`, or reject as soon as `signal` aborts (the work behind `p` is left to end on its own). */
function abortable<T>(p: Promise<T>, signal: AbortSignal): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    if (signal.aborted) {
      reject(abortReason(signal));
      return;
    }
    const onAbort = () => {
      reject(abortReason(signal));
    };
    signal.addEventListener("abort", onAbort, { once: true });
    p.then(resolve, reject).finally(() => {
      signal.removeEventListener("abort", onAbort);
    });
  });
}

/** `unavailable`: the record is there but its token cannot be read (keyring down, secret gone). */
type State = "ok" | "expired" | "invalid" | "error" | "unavailable";

interface Checked {
  /** Without a token when it could not be read; `login` is then the record alone. */
  login: LoginSummary & { token?: string };
  state: State;
  username: string;
  /** From the live check; undefined when the check did not run or did not say. */
  isAdmin?: boolean;
  problem?: string;
}

/**
 * Check each stored login against its Galaxy and say what is wrong with any that fail.
 *
 * Exits 1 when any login is not usable right now (expired, its token unreadable, refused by
 * Galaxy, or Galaxy did not answer) and 0 when all are: the command ran and reported on them, so
 * this is not one of the sysexits failures. `--json` exits 0 whenever it prints the list, as gh's
 * does, and the outcome is the `status` of each entry; a failure before there is a list (an
 * unreadable `hosts.yml`, a bad `--host`) or Ctrl-C (130) is still non-zero. Nothing stored is 77
 * (as `token` has it), except with `--json`, where an empty list is the answer.
 */
export async function statusCommand(opts: StatusOptions, deps: AuthDeps): Promise<number> {
  const err = (s: string) => {
    deps.io.stderr.write(`${s}\n`);
  };
  try {
    let summaries = await deps.store.list();
    // Every stored Galaxy, before `--host` narrows the list: with several, an operation that names
    // none is ambiguous, whichever one is being shown.
    const stored = summaries.map((s) => s.baseUrl);
    if (opts.host) {
      const baseUrl = await deps.resolveHost(opts.host);
      summaries = summaries.filter((s) => s.baseUrl === baseUrl);
      // The list leaves out a login this version cannot read (one a newer gxy wrote), and that is not
      // "not logged in": `get` says so, and that error ends the command, `--json` or not.
      if (summaries.length === 0) await deps.store.get(baseUrl);
      if (summaries.length === 0 && !opts.json) {
        err(
          `Not logged in to ${hostOf(baseUrl)}. Run: ${deps.commandName} auth login --host ${baseUrl}`,
        );
        return 77;
      }
    } else if (summaries.length === 0 && !opts.json) {
      err(`No stored logins. Run: ${deps.commandName} auth login --host <url>`);
      return 77;
    }

    const now = deps.now();
    /** One host's check; `undefined` when Ctrl-C ended it. */
    const check = async (summary: LoginSummary): Promise<Checked | undefined> => {
      // Each host on its own: one unreadable token must not hide the others.
      let login: (StoredLogin & { storage: LoginSummary["storage"] }) | null = null;
      let unreadable: string;
      try {
        login = await deps.store.get(summary.baseUrl);
        unreadable = "the token is not in the store";
      } catch (e) {
        unreadable = e instanceof Error ? e.message : String(e);
      }
      if (!login) {
        return {
          login: summary,
          state: "unavailable",
          username: summary.username,
          problem: `token unavailable: ${sanitize(unreadable)}`,
        };
      }
      // An expired token is not sent anywhere: the clock has already said no.
      const expiry = checkExpiry(login.expiresAt, now);
      if (expiry.state === "expired" || expiry.state === "unreadable") {
        return {
          login,
          state: "expired",
          username: login.username,
          problem: describeExpiry(login.expiresAt ?? "", now),
        };
      }
      const timeoutMs = opts.timeoutMs ?? CHECK_TIMEOUT_MS;
      const timeout = AbortSignal.timeout(timeoutMs);
      const signal = opts.signal ? AbortSignal.any([opts.signal, timeout]) : timeout;
      try {
        const who = await abortable(deps.whoami(login.baseUrl, login.token, signal), signal);
        return {
          login,
          state: "ok",
          username: who.username,
          ...(who.isAdmin !== undefined ? { isAdmin: who.isAdmin } : {}),
        };
      } catch (e) {
        if (opts.signal?.aborted) return undefined;
        if (timeout.aborted) {
          return {
            login,
            state: "error",
            username: login.username,
            problem: `No answer from ${hostOf(login.baseUrl)} within ${Math.round(timeoutMs / 1000)} s.`,
          };
        }
        // `not_logged_in` is Galaxy refusing the token. Anything else (no answer, a 5xx) says
        // nothing about the token, so it is an error and not "invalid".
        const invalid = e instanceof AuthError && e.kind === "not_logged_in";
        return {
          login,
          state: invalid ? "invalid" : "error",
          username: login.username,
          problem: sanitize(e instanceof Error ? e.message : String(e)),
        };
      }
    };
    // All at once, so N unreachable Galaxies cost one timeout and not N; `checked` keeps the order
    // of `summaries`, which is the order the report is printed in.
    const results = await Promise.all(summaries.map((summary) => check(summary)));
    if (results.includes(undefined)) return EX_INTERRUPTED;
    const checked = results as Checked[];
    // What an operation run now would use for each Galaxy: a login on the list is not always the key
    // in use (config.ts ranks `--api-key`, `--profile` and `GALAXY_API_KEY` above it, `.env` below),
    // and the flags or environment may send operations to another Galaxy altogether, in which case
    // no source is named for this one.
    const rows = checked.map((c) => {
      const { token } = c.login;
      const asked = deps.keySourceFor?.(
        c.login.baseUrl,
        token === undefined ? undefined : { ...c.login, token },
        stored,
      );
      const key: { id: KeySourceId | null; label: string } =
        asked ??
        (deps.env.GALAXY_API_KEY
          ? { id: "env", label: "GALAXY_API_KEY" }
          : token === undefined
            ? { id: null, label: "none (the stored token cannot be read)" }
            : { id: "stored", label: "stored login" });
      return { c, key };
    });

    if (opts.json) {
      const out = rows.map(({ c, key }) => ({
        base_url: c.login.baseUrl,
        username: sanitize(c.username),
        is_admin: c.isAdmin ?? null,
        storage: c.login.storage,
        method: c.login.method,
        scopes: c.login.scopes.map((s) => sanitize(s)),
        obtained_at: c.login.obtainedAt,
        expires_at: c.login.expiresAt,
        status: c.state,
        key_source: key.id,
        ...(c.problem ? { error: c.problem } : {}),
        ...(opts.showToken && c.login.token !== undefined ? { token: c.login.token } : {}),
      }));
      deps.io.stdout.write(`${JSON.stringify(out, null, 2)}\n`);
      return 0;
    }

    // Said first, because a green login below it can be about a key nothing is using.
    const outranking = [
      ...new Set(
        rows.filter((r) => r.key.id !== null && r.key.id !== "stored").map((r) => r.key.label),
      ),
    ];
    if (outranking.length)
      err(`${outranking.join(", ")} takes precedence over stored logins for operations.\n`);
    for (const { c, key } of rows) {
      const l = c.login;
      err(sanitize(hostOf(l.baseUrl)));
      if (c.state === "ok") err(`  ✓ Logged in as ${sanitize(c.username)}`);
      else if (c.state === "expired")
        err(`  ✗ Login ${c.problem}. Run: ${deps.commandName} auth login --host ${l.baseUrl}`);
      else if (c.state === "unavailable")
        err(`  ✗ ${c.problem}. Run: ${deps.commandName} auth login --host ${l.baseUrl}`);
      else err(`  ✗ ${c.problem}`);
      err(`  Storage  ${describeStorage(l.storage)}`);
      err("  Method   pasted API key");
      err("  Scopes   full access (API key)");
      const expiry = checkExpiry(l.expiresAt, now);
      if (expiry.state === "valid") {
        err(
          `  Expires  in ${formatTimeLeft(expiry.at.getTime() - now.getTime())} (${formatExpiry(expiry.at.toISOString())})`,
        );
      } else if (expiry.state === "none") {
        err("  Expires  no expiry");
      }
      // Masked on stderr always: the secret itself, when asked for, goes to stdout.
      err(`  Token    ${l.token === undefined ? "unavailable" : maskToken(l.token)}`);
      err(`  Key used by operations: ${sanitize(key.label)}`);
      if (c.isAdmin)
        err(
          `  ! This token carries full admin rights on ${hostOf(l.baseUrl)}. Treat it like a password.`,
        );
      if (l.expiresAt === null && l.token !== undefined)
        err(
          "  ! This token has no expiry: it works until you revoke it in Galaxy's user preferences.",
        );
    }
    if (opts.showToken) {
      for (const c of checked)
        if (c.login.token !== undefined)
          deps.io.stdout.write(`${c.login.baseUrl}\t${c.login.token}\n`);
    }
    return checked.some((c) => c.state !== "ok") ? 1 : 0;
  } catch (e) {
    return fail(deps, e);
  }
}
