/**
 * The types of the credential store: what a stored login is, and the interface a credential
 * store implements.
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
 * Why a Galaxy URL or the credential file was refused: `usage` is a URL the person can fix (it is
 * not a URL, or has credentials in it), `insecure` is plain http to a host other than this
 * machine, `store` is a `hosts.yml` (or a login in it) that cannot be used as it is.
 */
export type AuthErrorKind =
  /** Plain http to a host other than loopback. */
  | "insecure"
  /**
   * `hosts.yml` cannot be used as it is, and nothing was changed: it is corrupt, a symlink,
   * another user's, writable by others, or the login asked for was written by a newer gxy. The
   * message says which, and what to do about it.
   */
  | "store"
  /** A URL or argument the user can fix. */
  | "usage";

export class AuthError extends Error {
  constructor(
    readonly kind: AuthErrorKind,
    message: string,
  ) {
    super(message);
    this.name = "AuthError";
  }
}
