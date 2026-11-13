import { normalizeBaseUrl } from "../base-url";
import { InputCancelled, isRawCapable, readPiped, readSecret } from "../stdin";
import { AuthError, type AuthDeps, type BaseUrl, type StoredLogin, type WhoAmI } from "../types";
import { fail, hostOf, sanitize } from "./common";
import { abortReason } from "../abort";

export interface LoginOptions {
  host?: string;
  withToken?: boolean;
  insecureStorage?: boolean;
  /** Aborted on Ctrl-C by the caller (commands don't touch process signals). */
  signal?: AbortSignal;
}

/** Same Galaxy, however the URL was written; an unparsable one never matches. */
function sameGalaxy(a: string, b: string): boolean {
  try {
    return normalizeBaseUrl(a) === b;
  } catch {
    return false;
  }
}

/** 128 + SIGINT, which is what a shell reports for a process that Ctrl-C ended. */
const EX_INTERRUPTED = 130;

/** Settle with `p`, or reject as soon as `signal` aborts (the work behind `p` is left to finish on its own). */
function abortable<T>(p: Promise<T>, signal: AbortSignal): Promise<T> {
  if (signal.aborted) {
    p.catch(() => undefined);
    return Promise.reject(abortReason(signal));
  }
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => {
      reject(abortReason(signal));
    };
    signal.addEventListener("abort", onAbort, { once: true });
    p.then(resolve, reject).finally(() => {
      signal.removeEventListener("abort", onAbort);
    });
  });
}

/** Printed before the token is stored, so it shows even when storing is refused. */
function adminWarning(host: string, who: WhoAmI): string | undefined {
  return who.isAdmin
    ? `! This token carries full admin rights on ${host}. Treat it like a password.`
    : undefined;
}

function storedMessage(storage: "keyring" | "file"): string {
  const where = storage === "keyring" ? "the system keyring" : "a file readable only by you";
  return `✓ Token stored in ${where} (no expiry)`;
}

/**
 * Store an API key read from stdin (`--with-token`).
 *
 * The Galaxy comes only from `--host`, `GALAXY_URL` in the environment or the one stored login,
 * never `.env` or a profile, and the person is told where the login is going before a key is
 * read. `gh auth login --with-token` reads stdin the same way, and the key is never an argument:
 * it would sit in shell history and `ps`, which is why `--api-key` is deprecated.
 *
 * The key goes through `whoami` before it is stored, so a key that Galaxy does not accept is
 * never kept, and `whoami` runs the `/api/version` probe, so a key is only sent to something
 * that answers as a Galaxy.
 */
export async function loginCommand(opts: LoginOptions, deps: AuthDeps): Promise<number> {
  const err = (s: string) => {
    deps.io.stderr.write(`${s}\n`);
  };
  // One signal for the whole command: every step below either takes it or is raced against it.
  const signal = opts.signal ?? new AbortController().signal;
  try {
    if (!opts.withToken)
      throw new AuthError(
        "usage",
        `${deps.commandName} auth login needs --with-token: pipe your Galaxy API key in on stdin, ` +
          `\`printf %s "$KEY" | ${deps.commandName} auth login --host <url> --with-token\``,
      );
    // 1. Which Galaxy.
    const baseUrl: BaseUrl = await abortable(deps.resolveHost(opts.host, "login"), signal);
    const host = hostOf(baseUrl);
    // Login never takes its host from `.env` or a profile (see pickHost); say so when one disagrees.
    if (!opts.host) {
      const ignored = deps.ignoredLoginHost?.();
      if (ignored && !sameGalaxy(ignored.url, baseUrl)) {
        err(
          ignored.source === "dotenv"
            ? `Ignoring GALAXY_URL from .env (${ignored.url}): login only uses --host or GALAXY_URL from the environment.`
            : `Ignoring galaxy_url from the --profile (${ignored.url}): login only uses --host or GALAXY_URL from the environment.`,
        );
      }
    }
    // Before a key is read: the person sees where it is going.
    err(`Logging in to ${host}`);

    // Operations take `GALAXY_API_KEY` before any stored login (config.ts), so the login made here
    // would go unused. gh refuses to log in while `GH_TOKEN` is set; this warns and carries on.
    if (deps.env.GALAXY_API_KEY) {
      err(
        "! GALAXY_API_KEY is set and will take precedence over the login stored now. Unset it to use the stored login.",
      );
    }

    // 2. A key pasted on stdin, or typed at a terminal (without echo, so it stays off the screen
    // and out of scrollback).
    const pipeIn = `printf %s "$KEY" | ${deps.commandName} auth login --host ${baseUrl} --with-token`;
    let typed: string;
    // Whether to prompt follows stdin alone: a terminal on stdin is read in cooked mode, which
    // echoes the key and waits for Ctrl-D, whatever stderr is. The prompt goes to stderr, so when
    // that is redirected (`2>log`, `| tee`) there is nowhere to show it, and the key is refused
    // rather than asked for blind.
    const terminal = isRawCapable(deps.io.stdin) ? deps.io.stdin : undefined;
    const atTerminal = terminal !== undefined;
    if (terminal && !deps.io.interactive)
      throw new AuthError(
        "usage",
        `stdin is a terminal but stderr is not, so there is nowhere to show the key prompt. Pipe the key in: ${pipeIn}`,
      );
    if (terminal) {
      deps.io.stderr.write("Paste your Galaxy API key: ");
      typed = await readSecret(terminal, signal);
      deps.io.stderr.write("\n");
    } else {
      typed = await readPiped(deps.io.stdin, signal);
    }
    const raw = typed.trim();
    if (!raw)
      throw new AuthError(
        "usage",
        atTerminal
          ? `No API key entered. Paste it at the prompt, or pipe it in: ${pipeIn}`
          : `No API key on stdin. Pipe it in: ${pipeIn}`,
      );
    if (/[\r\n]/.test(raw))
      throw new AuthError("usage", "Expected a single-line API key on stdin, got several lines.");
    const who = await abortable(deps.whoami(baseUrl, raw, signal), signal);
    const login: StoredLogin = {
      baseUrl,
      username: who.username,
      ...(who.email ? { email: who.email } : {}),
      token: raw,
      method: "pasted",
      scopes: [],
      expiresAt: null,
      obtainedAt: deps.now().toISOString(),
    };
    // Last check before the write. The write itself can be interrupted too (a keyring unlock
    // prompt can last a minute), so the signal goes to `put`, which cancels rather than falling
    // back to the plaintext file.
    signal.throwIfAborted();
    const adminWarn = adminWarning(host, who);
    if (adminWarn) err(adminWarn);
    // An admin key is unscoped, so it is kept in the keyring or not at all: with no keyring,
    // login refuses unless `--insecure-storage` says a plaintext file is acceptable.
    const put = await deps.store.put(login, {
      insecureStorage: opts.insecureStorage,
      ...(who.isAdmin ? { requireKeyring: true } : {}),
      signal,
    });
    if (put.warning) err(`! ${put.warning}`);
    err(`✓ Logged in to ${host} as ${sanitize(who.username)}`);
    err(storedMessage(put.storage));
    err("! This token has no expiry: it works until you revoke it in Galaxy's user preferences.");
    return 0;
  } catch (e) {
    // A Ctrl-C at the key prompt arrives as a character in raw mode, not as SIGINT, hence
    // `InputCancelled`; either way it is 130 and not a failure message.
    if (signal.aborted || e instanceof InputCancelled) {
      err("Cancelled.");
      return EX_INTERRUPTED;
    }
    return fail(deps, e);
  }
}
