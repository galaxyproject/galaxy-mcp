import type { AuthDeps } from "../types";
import { fail, requireLogin } from "./common";

export interface TokenOptions {
  host?: string;
  json?: boolean;
}

/**
 * Print the stored token, and nothing else on stdout, so `$(gxy auth token)` is the token.
 *
 * Like `gh auth token` it prints with no terminal check and no confirmation: a refusal would
 * break the scripts that are what it is for, and those have no terminal.
 * `token_type` in the JSON says which header carries the value, `x-api-key`: that is what
 * galaxy-ops sends, and Galaxy's API also reads `Authorization: Bearer` when no key is given.
 */
export async function tokenCommand(opts: TokenOptions, deps: AuthDeps): Promise<number> {
  try {
    const login = await requireLogin(deps, opts.host);
    if (opts.json) {
      deps.io.stdout.write(
        `${JSON.stringify({ version: 1, base_url: login.baseUrl, token: login.token, token_type: "x-api-key", expires_at: login.expiresAt })}\n`,
      );
    } else {
      deps.io.stdout.write(`${login.token}\n`);
    }
    return 0;
  } catch (e) {
    return fail(deps, e);
  }
}
