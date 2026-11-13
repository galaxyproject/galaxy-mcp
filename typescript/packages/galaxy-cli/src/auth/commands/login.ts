import { createInterface } from "node:readline";
import { normalizeBaseUrl } from "../base-url";
import { InputCancelled, isRawCapable, readPiped, readSecret } from "../stdin";
import {
  AuthError,
  type AuthDeps,
  type AuthServer,
  type BaseUrl,
  type PutResult,
  type StoredLogin,
  type TokenGrant,
  type WhoAmI,
} from "../types";
import { fail, formatCountdown, formatExpiry, hostOf, sanitize } from "./common";
import { abortReason, withDeadline } from "../abort";
import { trustedVerificationUri } from "../verification-uri";

export interface LoginOptions {
  host?: string;
  scopes?: string[];
  /** `--no-browser` sets this false. */
  browser?: boolean;
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

/** Lines from stdin, one at a time; readline buffers what arrives between reads. */
function lineReader(stdin: NodeJS.ReadableStream) {
  const rl = createInterface({ input: stdin, terminal: false });
  const lines = rl[Symbol.asyncIterator]();
  return {
    async next(): Promise<string | null> {
      const r = await lines.next();
      return r.done ? null : r.value;
    },
    close: () => {
      rl.close();
    },
  };
}

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

/** Warnings a person should see after a fresh login is stored (the admin one came earlier). */
function tokenWarnings(expiresAt: string | null): string[] {
  return expiresAt === null
    ? ["! This token has no expiry: it works until you revoke it in Galaxy's user preferences."]
    : [];
}

/**
 * One wording and hint for every way a Galaxy can lack device login: 76 (EX_PROTOCOL), the code
 * for a server that does not speak what we need, and `--with-token` as the way in meanwhile.
 *
 * It is what a person on Galaxy 26.1 sees today, which publishes no OAuth metadata, and will
 * keep seeing until the device flow ships (galaxyproject/galaxy#23645).
 */
function noDeviceLogin(deps: AuthDeps, host: string, baseUrl: string, why: string): AuthError {
  return new AuthError(
    "unsupported",
    `${host} does not offer device login yet (${why}). Paste an API key instead: ${deps.commandName} auth login --host ${baseUrl} --with-token`,
  );
}

function storedMessage(storage: "keyring" | "file", expiresAt: string | null): string {
  const where = storage === "keyring" ? "the system keyring" : "a file readable only by you";
  return `✓ Token stored in ${where} (${expiresAt ? `expires ${formatExpiry(expiresAt)}` : "no expiry"})`;
}

/**
 * Take back a token the server issued but that cannot be kept, and say what happened either
 * way: the person may have to revoke it by hand. Returns whether it already said "Cancelled", so
 * the caller does not say it twice.
 *
 * `cancelled` is a Ctrl-C that landed after the token was issued. The revocation is its own
 * bounded call (`withDeadline`), not the login's signal, which is already aborted at that point:
 * sharing it would fail the revoke before it started and leave the token valid.
 */
async function revokeAfterFailure(
  server: AuthServer,
  token: string,
  host: string,
  err: (s: string) => void,
  cancelled: boolean,
): Promise<boolean> {
  if (!server.supportsRevocation) {
    err(
      cancelled
        ? `! Cancelled, but ${host} had already issued a token, and it cannot be revoked from here. Revoke it in Galaxy's user preferences.`
        : `! ${host} issued a token that could not be kept, and it cannot be revoked from here. Revoke it in Galaxy's user preferences.`,
    );
    return cancelled;
  }
  try {
    await withDeadline((signal) => server.revoke(token, signal));
    err(
      cancelled
        ? `Cancelled; the token ${host} had issued was revoked.`
        : `! The token ${host} issued could not be kept, so it was revoked.`,
    );
  } catch {
    err(
      cancelled
        ? `! Cancelled, but ${host} had issued a token and revoking it failed. Revoke it in Galaxy's user preferences.`
        : `! ${host} issued a token that could not be kept, and revoking it failed. Revoke it in Galaxy's user preferences.`,
    );
  }
  return cancelled;
}

/**
 * Log in by the device flow (RFC 8628), or with `--with-token` store a key from stdin.
 *
 * The Galaxy comes only from `--host`, `GALAXY_URL` in the environment or the one stored login,
 * never `.env` or a profile, and the person is told where the login is going before a key is
 * read or a code asked for. `gh auth login --with-token` reads stdin the same way, and the key is
 * never an argument: it would sit in shell history and `ps`, which is why `--api-key` is deprecated.
 *
 * Either way the token goes through `whoami` before it is stored, so a key that Galaxy does not
 * accept is never kept, and `whoami` runs the `/api/version` probe, so a token is only sent to
 * something that answers as a Galaxy.
 */
export async function loginCommand(opts: LoginOptions, deps: AuthDeps): Promise<number> {
  const err = (s: string) => {
    deps.io.stderr.write(`${s}\n`);
  };
  // One signal for the whole command: every step below either takes it or is raced against it.
  const signal = opts.signal ?? new AbortController().signal;
  let reader: ReturnType<typeof lineReader> | undefined;
  // Set when the revocation after a cancel has already printed "Cancelled".
  const said = { cancelled: false };
  try {
    // 1. Which Galaxy.
    let baseUrl: BaseUrl;
    if (opts.host) {
      baseUrl = await abortable(deps.resolveHost(opts.host, "login"), signal);
    } else {
      try {
        baseUrl = await abortable(deps.resolveHost(undefined, "login"), signal);
      } catch (e) {
        // Asked only when nothing at all names a Galaxy and a person is there to answer. With
        // `--with-token` stdin is for the key, so it cannot also be asked for a URL.
        if (
          !(e instanceof AuthError && e.kind === "usage") ||
          !deps.io.interactive ||
          opts.withToken
        )
          throw e;
        reader = lineReader(deps.io.stdin);
        deps.io.stderr.write("Galaxy URL (for example https://usegalaxy.org): ");
        const answer = await abortable(reader.next(), signal);
        if (!answer?.trim())
          throw new AuthError("usage", "No Galaxy URL given. Pass --host <url>.");
        baseUrl = normalizeBaseUrl(answer);
      }
    }
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
    // Before a key is read or a code is requested: the person sees where it is going.
    err(`Logging in to ${host}`);

    // Operations take `GALAXY_API_KEY` before any stored login (config.ts), so the login made here
    // would go unused. gh refuses to log in while `GH_TOKEN` is set; this warns and carries on.
    if (deps.env.GALAXY_API_KEY) {
      err(
        "! GALAXY_API_KEY is set and will take precedence over the login stored now. Unset it to use the stored login.",
      );
    }

    // 2a. A key pasted on stdin, or typed at a terminal (without echo, so it stays off the screen
    // and out of scrollback).
    if (opts.withToken) {
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
      err(storedMessage(put.storage, null));
      for (const w of tokenWarnings(null)) err(w);
      return 0;
    }

    // 2b. The device flow.
    let server: AuthServer;
    try {
      server = await abortable(deps.discover(baseUrl, signal), signal);
    } catch (e) {
      // Discovery says "unsupported" only when there is no usable metadata (404, not JSON), which
      // is what Galaxy 26.1 gives; `noDeviceLogin` points at `--with-token`.
      if (e instanceof AuthError && e.kind === "unsupported")
        throw noDeviceLogin(deps, host, baseUrl, "it publishes no OAuth metadata");
      throw e;
    }
    if (!server.supportsDeviceFlow)
      throw noDeviceLogin(deps, host, baseUrl, "it may be switched off by its admins");
    const scopes = opts.scopes ?? [];
    // A cancel during discovery must not create a code on the server.
    signal.throwIfAborted();
    const pending = await abortable(
      server.startDeviceLogin(scopes.length ? scopes : undefined, signal),
      signal,
    );

    // `startDeviceLogin` checks this too; this is for any other `AuthServer`. The normalised URL is
    // what is printed and opened: the raw string can hold a tab or newline that parsing drops.
    const verificationUri = trustedVerificationUri(pending.verificationUri, baseUrl);
    if (verificationUri === undefined) {
      throw new AuthError(
        "protocol",
        `${host} asked for the code to be entered at ${sanitize(pending.verificationUri)}, which is not an https page on ${host}. ` +
          "That could send you to another site, so the login was stopped. Paste an API key instead: " +
          `${deps.commandName} auth login --host ${baseUrl} --with-token`,
      );
    }

    err(`Requesting access to ${host}`);
    err(`  Scopes  ${scopes.length ? sanitize(scopes.join(" ")) : "server default"}`);
    err("");
    // RFC 8628 §5.4 (remote phishing): a person can be talked into approving a code they never
    // started, so the terminal that did start one says to approve only that. Only the user code is
    // printed; the device code never leaves `waitForApproval`.
    err("! Only approve this code if you ran this command yourself.");
    err(`  One-time code: ${sanitize(pending.userCode)}`);
    err(`  Enter it at ${sanitize(verificationUri)}`);

    // As `gh auth login` does: offer on a terminal and wait for Enter, and print the URL and nothing
    // more when not interactive, so a script or an agent never has a browser opened for it.
    const canOpen =
      opts.browser !== false && deps.io.interactive && (deps.canOpenBrowser?.() ?? true);
    if (canOpen) {
      err("Press Enter to open that page in your browser…");
      reader ??= lineReader(deps.io.stdin);
      // Polling starts at once: approving on another device must not wait for Enter.
      void reader.next().then(async (line) => {
        if (line === null || signal.aborted) return;
        if (!(await deps.openBrowser(verificationUri)))
          err("Could not open a browser. Open the page yourself.");
      });
    }
    err(
      `Waiting for approval (code expires in ${formatCountdown(pending.expiresAt.getTime() - deps.now().getTime())})…`,
    );
    const grant: TokenGrant = await pending.waitForApproval(signal);

    // The server has issued a token. If it cannot be checked or kept, nothing local points at
    // it, so it would stay valid (an admin's too) until it expires: take it back.
    let who: WhoAmI;
    let put: PutResult;
    let expiresAt: string | null;
    try {
      who = await abortable(deps.whoami(baseUrl, grant.accessToken, signal), signal);
      const now = deps.now();
      expiresAt =
        grant.expiresIn !== undefined
          ? new Date(now.getTime() + grant.expiresIn * 1000).toISOString()
          : null;
      signal.throwIfAborted();
      const adminWarn = adminWarning(host, who);
      if (adminWarn) err(adminWarn);
      // The same admin rule as for a pasted key, above.
      put = await deps.store.put(
        {
          baseUrl,
          username: who.username,
          ...(who.email ? { email: who.email } : {}),
          token: grant.accessToken,
          method: "device",
          scopes: grant.scopes,
          expiresAt,
          obtainedAt: now.toISOString(),
        },
        {
          insecureStorage: opts.insecureStorage,
          ...(who.isAdmin ? { requireKeyring: true } : {}),
          signal,
        },
      );
    } catch (e) {
      said.cancelled = await revokeAfterFailure(
        server,
        grant.accessToken,
        host,
        err,
        signal.aborted,
      );
      throw e;
    }
    if (put.warning) err(`! ${put.warning}`);
    err(`✓ Logged in to ${host} as ${sanitize(who.username)}`);
    err(storedMessage(put.storage, expiresAt));
    for (const w of tokenWarnings(expiresAt)) err(w);
    return 0;
  } catch (e) {
    // A Ctrl-C at the key prompt arrives as a character in raw mode, not as SIGINT, hence
    // `InputCancelled`; either way it is 130 and not a failure message.
    if (signal.aborted || e instanceof InputCancelled) {
      if (!said.cancelled) err("Cancelled.");
      return EX_INTERRUPTED;
    }
    return fail(deps, e);
  } finally {
    reader?.close();
  }
}
