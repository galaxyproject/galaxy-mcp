import { withDeadline } from "../abort";
import { checkExpiry } from "../expiry";
import type { AuthDeps, StoredLogin } from "../types";
import { fail, formatExpiry, hostOf, sanitize } from "./common";

export interface LogoutOptions {
  host?: string;
}

/**
 * Revoke on the server first (device logins only), then delete locally whatever happens.
 *
 * The order is because revoking needs the token, and a failed revocation must not leave the
 * login stored: a person who ran `logout` wants it gone from here, and is told when the server
 * still honours the token and until when. A pasted key is only forgotten, not revoked: it may be
 * in use elsewhere (a script, another machine), so the output says it still works until it is
 * revoked, and where to do that.
 *
 * Revoking is bounded (`withDeadline`, discovery included): logout is what a person runs when
 * the server is down or the network is wrong, and a hang there would delay the local removal they
 * asked for until the network gave up on its own, minutes later. A token that has already expired
 * is not revoked, since there is nothing left to take back.
 */
export async function logoutCommand(opts: LogoutOptions, deps: AuthDeps): Promise<number> {
  const err = (s: string) => {
    deps.io.stderr.write(`${s}\n`);
  };
  try {
    const baseUrl = await deps.resolveHost(opts.host);
    const host = hostOf(baseUrl);
    // The record (no secret) decides whether there is anything to remove. Reading the token is
    // only needed to revoke it, and must not stand between the person and removing the record:
    // the keyring may be down or the secret gone, which is when cleaning up matters most.
    const summary = (await deps.store.list()).find((s) => s.baseUrl === baseUrl);
    if (!summary) {
      // The list leaves out a login this version cannot read (one a newer gxy wrote), and that is
      // not "nothing to remove": `get` says so, and it ends here with its error rather than a 0.
      await deps.store.get(baseUrl);
      err(`Not logged in to ${host}; nothing to remove.`);
      return 0;
    }
    let login: StoredLogin | null = null;
    let unreadable: string | null = null;
    try {
      login = await deps.store.get(baseUrl);
      if (!login) unreadable = "the token is not in the store";
    } catch (e) {
      unreadable = sanitize(e instanceof Error ? e.message : String(e));
    }

    const expiry = summary.expiresAt ? checkExpiry(summary.expiresAt, deps.now()) : undefined;
    const alreadyExpired =
      expiry?.state === "expired" && expiry.at.getTime() <= deps.now().getTime();

    let revoked = false;
    let revokeProblem: string | null = null;
    if (summary.method === "device" && alreadyExpired) {
      // Nothing to take back, so nothing to try and nothing to warn about.
    } else if (!login) {
      revokeProblem = `the stored token could not be read: ${unreadable ?? "no reason given"}`;
    } else if (login.method === "device") {
      // Only a device login has a revocation call to make (RFC 7009). Discovery runs again here
      // because the store keeps no endpoints, only the login.
      try {
        const problem = await withDeadline(async (signal) => {
          const server = await deps.discover(baseUrl, signal);
          if (!server.supportsRevocation) return `${host} does not offer token revocation`;
          await server.revoke(login.token, signal);
          return null;
        });
        revokeProblem = problem;
        revoked = problem === null;
      } catch (e) {
        revokeProblem = sanitize(e instanceof Error ? e.message : String(e));
      }
    }

    // Local removal is attempted whatever the server said, and its own failure is reported
    // too: a secret that may remain in the keyring is something the person must know.
    let deleteProblem: string | null = null;
    try {
      await deps.store.delete(baseUrl);
    } catch (e) {
      deleteProblem = sanitize(e instanceof Error ? e.message : String(e), { keepNewlines: true });
    }

    // The removal line says what happened to the stored login: a tick only when it is gone.
    const removal = (what: string) =>
      deleteProblem
        ? `! Could not remove the stored ${what} for ${host}.`
        : `✓ Removed the stored ${what} for ${host}`;
    if (summary.method === "pasted") {
      err(removal("API key"));
      err(
        `! The key itself was not revoked, as it may be in use elsewhere. It still works until you revoke it in Galaxy's user preferences: ${baseUrl}/user/api_key`,
      );
    } else if (revoked) {
      err(`✓ Revoked the token on ${host}`);
      err(removal("login"));
    } else {
      err(removal("login"));
      if (!alreadyExpired) {
        const until = summary.expiresAt
          ? `until it expires (${formatExpiry(summary.expiresAt)})`
          : "until it is revoked in Galaxy";
        err(
          `! Could not revoke the token on the server (${revokeProblem ?? "no reason given"}). It stays valid ${until}.`,
        );
      }
    }
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
