import type { AuthDeps } from "../types";
import { fail, hostOf, sanitize } from "./common";

export interface LogoutOptions {
  host?: string;
}

/**
 * Delete the stored login for one Galaxy.
 *
 * A pasted key is only forgotten, not revoked: it may be in use elsewhere (a script, another
 * machine), so the output says it still works until it is revoked, and where to do that.
 */
export async function logoutCommand(opts: LogoutOptions, deps: AuthDeps): Promise<number> {
  const err = (s: string) => {
    deps.io.stderr.write(`${s}\n`);
  };
  try {
    const baseUrl = await deps.resolveHost(opts.host);
    const host = hostOf(baseUrl);
    // The record (no secret) decides whether there is anything to remove. The token is never
    // read: the keyring may be down or the secret gone, which is when cleaning up matters most.
    const summary = (await deps.store.list()).find((s) => s.baseUrl === baseUrl);
    if (!summary) {
      // The list leaves out a login this version cannot read (one a newer gxy wrote), and that is
      // not "nothing to remove": `get` says so, and it ends here with its error rather than a 0.
      await deps.store.get(baseUrl);
      err(`Not logged in to ${host}; nothing to remove.`);
      return 0;
    }

    // A failure to remove is reported too: a secret that may remain in the keyring is something
    // the person must know.
    let deleteProblem: string | null = null;
    try {
      await deps.store.delete(baseUrl);
    } catch (e) {
      deleteProblem = sanitize(e instanceof Error ? e.message : String(e), { keepNewlines: true });
    }

    // The removal line says what happened to the stored login: a tick only when it is gone.
    err(
      deleteProblem
        ? `! Could not remove the stored API key for ${host}.`
        : `✓ Removed the stored API key for ${host}`,
    );
    err(
      `! The key itself was not revoked, as it may be in use elsewhere. It still works until you revoke it in Galaxy's user preferences: ${baseUrl}/user/api_key`,
    );
    if (deleteProblem) {
      // The secret may still be in the keyring, so this is a failure (1) even though the record
      // is gone.
      err(`! ${deleteProblem}`);
      return 1;
    }
    return 0;
  } catch (e) {
    return fail(deps, e);
  }
}
