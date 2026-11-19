import type { GalaxyErrorKind } from "@galaxyproject/galaxy-ops";

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
  // "permission denied": the host's scope for this run does not allow the target. Shares the
  // code with a bad key because both say the caller may not do this, and neither is retryable.
  out_of_scope: EX_NOPERM,
  // The caller passed an id of one kind where another belongs: an argument to fix, like any
  // other usage error.
  handle_kind_mismatch: EX_USAGE,
  unknown: EX_SOFTWARE,
};

export function exitCodeFor(kind?: GalaxyErrorKind): number {
  if (!kind) return 0;
  return MAP[kind] ?? EX_SOFTWARE;
}
