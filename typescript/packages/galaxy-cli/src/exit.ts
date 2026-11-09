import type { GalaxyErrorKind } from "@galaxyproject/galaxy-ops";
import type { AuthErrorKind } from "./auth/types";

export const EX_USAGE = 64;
export const EX_DATAERR = 65;
export const EX_NOINPUT = 66;
export const EX_UNAVAILABLE = 69;
export const EX_SOFTWARE = 70;
export const EX_PROTOCOL = 76;
export const EX_NOPERM = 77;

const MAP: Record<GalaxyErrorKind, number> = {
  auth: EX_NOPERM,
  not_found: EX_NOINPUT,
  validation: EX_USAGE,
  connection: EX_UNAVAILABLE,
  // "remote error in protocol": the server we reached does not speak a new enough Galaxy.
  version: EX_PROTOCOL,
  tool_request_rejected: EX_DATAERR,
  job_failed: EX_SOFTWARE,
  unknown: EX_SOFTWARE,
};

export function exitCodeFor(kind?: GalaxyErrorKind): number {
  if (!kind) return 0;
  return MAP[kind] ?? EX_SOFTWARE;
}

/**
 * Exit codes for `auth` failures, by kind (docs: "Exit codes"), the same sysexits an operation
 * gets for the same kind of trouble, so a script handles both alike. Ctrl-C (130) and `status`
 * finding a bad login (1) are decided in the commands, not here.
 */
const AUTH_MAP: Record<AuthErrorKind, number> = {
  // The person can fix the command line; with several stored logins, by naming one.
  usage: EX_USAGE,
  ambiguous_host: EX_USAGE,
  // As `connection` for an operation: no answer from the Galaxy.
  network: EX_UNAVAILABLE,
  // As `version`: a Galaxy that cannot be used for login as it is. `unsupported` is one with no
  // device flow; `insecure` is one that cannot be reached safely: plain http to a host that is not
  // this machine, which is refused before anything is sent, or an issuer or endpoints that are not
  // itself. That is 76 and not 64 even for a typed `--host http://...`: it is the Galaxy's address
  // that is refused, whoever wrote it, and one code covers it wherever the address came from.
  unsupported: EX_PROTOCOL,
  protocol: EX_PROTOCOL,
  insecure: EX_PROTOCOL,
  // As `auth` for an operation: there is no credential Galaxy would take, whether none is stored,
  // it ran out, an admin key may not go in the plaintext file, or the person said no on the
  // approval page.
  not_logged_in: EX_NOPERM,
  denied: EX_NOPERM,
  // `hosts.yml` is the input that cannot be used (corrupt, loose, a link, another user's, or a login
  // a newer gxy wrote), so EX_DATAERR, as for a tool request whose input is rejected. Not 77: a
  // script reads that as "log in", and logging in meets the same refusal. Not 70: that says gxy has
  // a bug, and the message says what to do.
  store: EX_DATAERR,
  expired: EX_NOPERM,
};

export function exitCodeForAuthError(kind: AuthErrorKind): number {
  return AUTH_MAP[kind] ?? EX_SOFTWARE;
}
