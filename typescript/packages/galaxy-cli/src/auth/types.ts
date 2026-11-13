/**
 * The seams between the parts of `auth`: where a login is kept, how one is obtained
 * from Galaxy, and what the commands are given to do their work.
 *
 * Every command takes an `AuthDeps` rather than reaching for the filesystem, the
 * network, the clock or the terminal itself, so each one can be tested with fakes and
 * the real implementations are wired together in one place (`deps.ts`).
 */

/**
 * A Galaxy's base URL in the form logins are keyed by: scheme, host, port and path
 * prefix, without a trailing slash or query (`https://usegalaxy.org`,
 * `https://example.org/galaxy`). Produced by `normalizeBaseUrl`; never compared raw.
 *
 * The brand is there so the compiler refuses a plain string where a key is wanted: a URL that
 * skipped normalisation (`https://X/api/`) would otherwise become a second login for the same
 * Galaxy, or miss the one that exists.
 */
export type BaseUrl = string & { readonly __brand: "BaseUrl" };

/** Where a token is kept: `file` is `hosts.yml`, readable only by the owner. */
export type StorageKind = "file";

/** How a token was obtained. */
type LoginMethod = "pasted";

/** One login to one Galaxy, as stored. */
export interface StoredLogin {
  baseUrl: BaseUrl;
  /** Galaxy username from `/api/users/current`, at login time. */
  username: string;
  email?: string;
  token: string;
  method: LoginMethod;
  /** Scopes the server granted (empty for a pasted API key, which carries the user's full power). */
  scopes: string[];
  /** ISO 8601, or null when the server gave no expiry (always null for a pasted API key). */
  expiresAt: string | null;
  /** ISO 8601. */
  obtainedAt: string;
}

/** A stored login without its secret, safe to print or list. */
export type LoginSummary = Omit<StoredLogin, "token"> & { storage: StorageKind };

export interface CredentialStore {
  /** The login for this Galaxy, or null if there is no record of one. */
  get(baseUrl: BaseUrl): Promise<(StoredLogin & { storage: StorageKind }) | null>;
  /** Store or replace the login for `login.baseUrl`. */
  put(login: StoredLogin): Promise<void>;
  /** Remove the login for this Galaxy. Returns false if there was none. */
  delete(baseUrl: BaseUrl): Promise<boolean>;
  /** Every stored login, without tokens, sorted by base URL. */
  list(): Promise<LoginSummary[]>;
}

/**
 * Why an auth command failed. `exitCodeForAuthError` (`exit.ts`) turns each into a sysexits.h
 * code: `usage` and `ambiguous_host` are 64 (EX_USAGE: the command line is wrong or does not
 * say enough), `store` is 65 (EX_DATAERR: the credential file cannot be used as it is), `network`
 * is 69 (EX_UNAVAILABLE), `protocol` and `insecure` are 76 (EX_PROTOCOL: the address, or what its
 * server supplied, is not something this can use; plain http is 76 too, though the person typed
 * it, see `exitCodeForAuthError`), and `not_logged_in` and `denied` are 77 (EX_NOPERM: there is no
 * permission to do this).
 */
export type AuthErrorKind =
  /** An admin key was refused: it goes in the plaintext file only when `--insecure-storage` says so. */
  | "denied"
  /** No stored login for this Galaxy, or it has expired, or Galaxy rejected the token. */
  | "not_logged_in"
  /** Several stored logins and no way to tell which Galaxy was meant. */
  | "ambiguous_host"
  /** Plain http to a host other than loopback, or an address that is not a Galaxy. */
  | "insecure"
  /**
   * `hosts.yml` cannot be used as it is, and nothing was changed: it is corrupt, a symlink,
   * another user's, writable by others, or the login asked for was written by a newer gxy. The
   * message says which, and what to do about it.
   */
  | "store"
  /** The server answered, but not as a Galaxy does. */
  | "protocol"
  /** The server could not be reached. */
  | "network"
  /** A flag or argument the user can fix. */
  | "usage";

export class AuthError extends Error {
  constructor(
    readonly kind: AuthErrorKind,
    message: string,
    options?: { cause?: unknown },
  ) {
    super(message, options);
    this.name = "AuthError";
  }
}

/** The user Galaxy says a token belongs to. */
export interface WhoAmI {
  username: string;
  email?: string;
  /** Galaxy's `is_admin`. An admin's token carries full, unscoped admin power. */
  isAdmin?: boolean;
}

interface AuthIo {
  stdin: NodeJS.ReadableStream;
  /** Machine-readable output only: a token, a JSON object. */
  stdout: { write(s: string): void };
  /** Everything meant for a person. */
  stderr: { write(s: string): void };
  /** stdin and stderr are both terminals, so it is safe to prompt. */
  interactive: boolean;
}

export interface AuthDeps {
  store: CredentialStore;
  /**
   * Ask Galaxy whose token this is (`GET /api/users/current` through galaxy-ops' `get_user`).
   * Rejects with `AuthError("not_logged_in")` when Galaxy refuses the token or answers with
   * the anonymous user (no username), which is what it does for a session without a valid key.
   */
  whoami(baseUrl: BaseUrl, token: string, signal?: AbortSignal): Promise<WhoAmI>;
  /**
   * Which Galaxy a command means: `--host`/`--url`, the `--profile` URL, `GALAXY_URL` (env, then
   * `.env`), then the only stored login (see `pickHost`). With `purpose: "login"` the
   * `.env` and profile are never consulted, so a repository cannot choose where a key goes.
   * Rejects with `AuthError("ambiguous_host")` listing the stored hosts when several are stored
   * and none was named, and `AuthError("usage")` when none is stored.
   */
  resolveHost(explicit?: string, purpose?: "login"): Promise<BaseUrl>;
  /**
   * The Galaxy named by `.env` (`GALAXY_URL`) or the selected profile, which `login` ignores by
   * design; lets it say so when that differs from the host it uses. Absent means none.
   */
  ignoredLoginHost?(): { source: "dotenv" | "profile"; url: string } | undefined;
  now(): Date;
  env: Record<string, string | undefined>;
  io: AuthIo;
  /** The name the user typed (`gxy` or `galaxy-cli`), for messages like "run gxy auth login". */
  commandName: string;
}
