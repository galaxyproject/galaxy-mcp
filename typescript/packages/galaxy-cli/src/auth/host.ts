import { normalizeBaseUrl } from "./base-url";
import { AuthError, type BaseUrl, type CredentialStore } from "./types";

/** Nothing named and nothing stored. A `usage` error for `login` (which then asks); `requireLogin` reports it as not logged in. */
export class NoHostError extends AuthError {
  constructor() {
    super("usage", "No Galaxy given and none logged in. Pass --host <url>.");
    this.name = "NoHostError";
  }
}

/**
 * `login` with no Galaxy named. A `usage` error. It never falls back to the only stored login:
 * a login takes a fresh key, and with one stored login for A, `printf %s "$KEY_FOR_B" | gxy auth
 * login --with-token` would send B's key to A.
 */
export class NoLoginHostError extends AuthError {
  constructor() {
    super(
      "usage",
      "No Galaxy named. Pass --host <url> (or --url), or set GALAXY_URL: login never picks a stored login, so a key only goes to the Galaxy you name.",
    );
    this.name = "NoLoginHostError";
  }
}

export type HostSource = "flag" | "env" | "dotenv" | "profile" | "stored";

interface PickHostInput {
  /** `--host` or `--url`. */
  flag?: string;
  env: Record<string, string | undefined>;
  /** Values from `.env` in the working directory. */
  dotenv: Record<string, string | undefined>;
  /** `galaxy_url` of the `--profile` planemo profile. */
  profileUrl?: string;
  /** Base URLs of every stored login. */
  storedBaseUrls: readonly BaseUrl[];
  /**
   * `login` sends a key, or asks a person to approve a code, for the host it picks, so it never
   * takes one from files in the working directory or the home directory (`.env`, a profile):
   * a repository must not be able to choose where your credentials go. Everything else
   * (reading a stored token, running an operation) only uses a host that must match a
   * stored login to get a token, so the files are safe there. For the same reason `login` never
   * falls back to the only stored login either (`NoLoginHostError`): the token it gets is a new
   * one, and need not belong to that Galaxy.
   *
   * The Python galaxy-mcp server holds to the same line from the other side: its
   * `GALAXY_API_KEY` is only used with the `GALAXY_URL` it was configured with.
   */
  forLogin?: boolean;
}

/**
 * The one rule for which Galaxy is meant, shared by `gxy auth *` and by operations, so they
 * cannot pick different ones: `--host`/`--url`, then the `--profile` URL, then `GALAXY_URL` in the
 * environment, then in `.env`, then the only stored login. The command line beats the
 * environment, which beats files, and a profile is a pair: its key goes to its own Galaxy, not to
 * whatever `GALAXY_URL` or a `.env` says. `login` (`forLogin`) takes the flag and the environment
 * only, with no stored-login fallback. Returns the URL as named (not normalised: operations keep
 * an `http://` URL that `auth` would refuse).
 */
export function pickHost(input: PickHostInput): { url: string; source: HostSource } {
  const named: [HostSource, string | undefined][] = input.forLogin
    ? [
        ["flag", input.flag],
        ["env", input.env.GALAXY_URL],
      ]
    : [
        ["flag", input.flag],
        ["profile", input.profileUrl],
        ["env", input.env.GALAXY_URL],
        ["dotenv", input.dotenv.GALAXY_URL],
      ];
  for (const [source, url] of named) if (url) return { url, source };
  if (input.forLogin) throw new NoLoginHostError();
  const stored = input.storedBaseUrls;
  const [only] = stored;
  if (stored.length === 1 && only !== undefined) return { url: only, source: "stored" };
  if (stored.length === 0) throw new NoHostError();
  throw new AuthError(
    "ambiguous_host",
    `Logged in to several Galaxies; say which with --host (or --url, or GALAXY_URL):\n${stored.map((u) => `  ${u}`).join("\n")}`,
  );
}

interface ResolveHostInput extends Omit<PickHostInput, "flag" | "storedBaseUrls"> {
  /** `--host` (or the global `--url`). */
  explicit?: string;
  store: Pick<CredentialStore, "list">;
}

/** `pickHost` for `gxy auth *`: lists the stored logins (only if needed) and normalises the result. */
export async function resolveHost(input: ResolveHostInput): Promise<BaseUrl> {
  const { explicit, store, ...rest } = input;
  const needsStore =
    !input.forLogin &&
    !(explicit || input.env.GALAXY_URL || input.dotenv.GALAXY_URL || input.profileUrl);
  const storedBaseUrls = needsStore ? (await store.list()).map((s) => s.baseUrl) : [];
  return normalizeBaseUrl(pickHost({ ...rest, flag: explicit, storedBaseUrls }).url);
}
