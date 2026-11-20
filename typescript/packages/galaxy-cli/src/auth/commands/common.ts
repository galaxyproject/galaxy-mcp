import { exitCodeForAuthError, EX_SOFTWARE } from "../../exit";
import { describeExpiry, isExpired } from "../expiry";
import { NoHostError } from "../host";
import { wasAnswered } from "../network";
import { sanitize } from "../sanitize";
import { AuthError, type AuthDeps, type BaseUrl, type StoredLogin } from "../types";

export { sanitize };

const PROXY_VARS = ["HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY"] as const;

/**
 * Node's `fetch` only honours a proxy variable when `NODE_USE_ENV_PROXY=1` was set before the
 * process started, so with one set and the flag off a failure to reach the server is probably the
 * proxy being ignored (not one where the server answered: `markAnswered`). Names the variable, never its value: a proxy URL can carry credentials.
 */
function proxyHint(env: AuthDeps["env"]): string | undefined {
  if (env.NODE_USE_ENV_PROXY === "1") return undefined;
  const name = PROXY_VARS.find((v) => env[v]);
  return (
    name &&
    `${name} is set, but Node ignores it unless NODE_USE_ENV_PROXY=1 is also set (Node 22.21+ or 24+).`
  );
}

/**
 * Print a failure for a person and map it to an exit code. Never includes a token: messages
 * are ours. They can still quote the server (a whoami failure), so
 * terminal control characters are stripped on the way out, or a Galaxy could rewrite the
 * terminal through an error text.
 *
 * An `AuthError` maps by its kind (`exitCodeForAuthError`). Anything else is not something the
 * person typed or a server did, so it is 70 (EX_SOFTWARE).
 */
export function fail(deps: AuthDeps, e: unknown): number {
  if (e instanceof AuthError) {
    deps.io.stderr.write(`${sanitize(e.message, { keepNewlines: true })}\n`);
    const hint = e.kind === "network" && !wasAnswered(e) ? proxyHint(deps.env) : undefined;
    if (hint) deps.io.stderr.write(`${sanitize(hint)}\n`);
    return exitCodeForAuthError(e.kind);
  }
  deps.io.stderr.write(
    `Unexpected error: ${sanitize(e instanceof Error ? e.message : String(e), { keepNewlines: true })}\n`,
  );
  return EX_SOFTWARE;
}

/** The host and path prefix, as shown: two Galaxies on one host are two logins, so the prefix stays. */
export function hostOf(baseUrl: string): string {
  try {
    return new URL(baseUrl).host + new URL(baseUrl).pathname.replace(/\/+$/, "");
  } catch {
    return baseUrl;
  }
}

/**
 * The stored, unexpired login for the host a command means, or an AuthError that says
 * how to log in. No network: `token` is called by scripts, so it answers from the store.
 */
export async function requireLogin(deps: AuthDeps, explicitHost?: string): Promise<StoredLogin> {
  let baseUrl: BaseUrl;
  try {
    baseUrl = await deps.resolveHost(explicitHost);
  } catch (e) {
    // Nothing stored and nothing named is "not logged in" here (77), not a usage error: the
    // caller is a helper that needs a token, and the fix is to log in.
    if (e instanceof NoHostError)
      throw new AuthError(
        "not_logged_in",
        `Not logged in; run ${deps.commandName} auth login --host <url>`,
        { cause: e },
      );
    throw e;
  }
  return requireLoginAt(deps, baseUrl);
}

/** `requireLogin` for a Galaxy the caller has already settled on. */
async function requireLoginAt(deps: AuthDeps, baseUrl: BaseUrl): Promise<StoredLogin> {
  const login = await deps.store.get(baseUrl);
  const hint = `${deps.commandName} auth login --host ${baseUrl}`;
  if (!login) throw new AuthError("not_logged_in", `Not logged in to ${baseUrl}. Run: ${hint}`);
  if (login.expiresAt !== null && isExpired(login.expiresAt, deps.now())) {
    throw new AuthError(
      "not_logged_in",
      `Your login to ${baseUrl} ${describeExpiry(login.expiresAt, deps.now())}. Run: ${hint}`,
    );
  }
  return login;
}
