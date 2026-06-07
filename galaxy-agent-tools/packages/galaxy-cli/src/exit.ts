import type { GalaxyErrorKind } from "@galaxyproject/galaxy-ops";

export const EX_USAGE = 64;
export const EX_DATAERR = 65;
export const EX_NOINPUT = 66;
export const EX_UNAVAILABLE = 69;
export const EX_SOFTWARE = 70;
export const EX_NOPERM = 77;

const MAP: Record<GalaxyErrorKind, number> = {
  auth: EX_NOPERM,
  not_found: EX_NOINPUT,
  connection: EX_UNAVAILABLE,
  tool_request_rejected: EX_DATAERR,
  job_failed: EX_SOFTWARE,
  unknown: EX_SOFTWARE,
};

export function exitCodeFor(kind?: GalaxyErrorKind): number {
  if (!kind) return 0;
  return MAP[kind] ?? EX_SOFTWARE;
}
