import { sanitize } from "./sanitize";

/**
 * A login counts as expired this long before `expiresAt`, so a token that passes the check
 * here still has time to do its job. `auth token` checks once and hands the
 * token over, and the server sees it later: without the margin a token with a second left
 * would be handed out as good and refused on arrival.
 */
export const EXPIRY_MARGIN_MS = 60_000;

type Expiry =
  | { state: "none" }
  | { state: "valid"; at: Date }
  /** Within the margin or past. */
  | { state: "expired"; at: Date }
  /** `expires_at` is not a time (a hand-edited or corrupt `hosts.yml`). Never trusted as "no expiry". */
  | { state: "unreadable" };

/**
 * Classify a stored expiry. `null` or absent is `none` (a pasted API key carries no expiry),
 * but a string that is not a date is `unreadable`, not `none`: a damaged record must fail
 * closed, or a typo in `hosts.yml` would turn a login that should have ended into one that
 * never does.
 */
export function checkExpiry(expiresAt: string | null | undefined, now: Date): Expiry {
  if (expiresAt === null || expiresAt === undefined) return { state: "none" };
  const ms = Date.parse(expiresAt);
  if (Number.isNaN(ms)) return { state: "unreadable" };
  const at = new Date(ms);
  return { state: ms - EXPIRY_MARGIN_MS <= now.getTime() ? "expired" : "valid", at };
}

/** True when the login must not be used: expired, about to expire, or with an unreadable expiry. */
export function isExpired(expiresAt: string | null | undefined, now: Date): boolean {
  const s = checkExpiry(expiresAt, now).state;
  return s === "expired" || s === "unreadable";
}

/** A stored value quoted back sanitised and cut short. It is typed `string` but comes from a file a person may have edited, so it is not trusted to be one. */
const quoted = (value: unknown): string => sanitize(String(value)).slice(0, 40);

/**
 * "2026-11-06 18:42 UTC"; never throws. An unparseable value is quoted back sanitised and cut
 * to 40 characters, because a hand-edited `hosts.yml` can hold anything, and this is printed.
 */
export function formatExpiry(iso: string): string {
  const ms = Date.parse(iso);
  if (Number.isNaN(ms)) return `an unreadable time (${quoted(iso)})`;
  return `${new Date(ms).toISOString().slice(0, 16).replace("T", " ")} UTC`;
}

/**
 * Why a login cannot be used, as a clause for the end of a sentence: "expired at ...", "is about
 * to expire (at ...)" for one inside the margin, or "has an unreadable expiry time ...".
 */
export function describeExpiry(expiresAt: string, now: Date): string {
  const e = checkExpiry(expiresAt, now);
  if (e.state === "unreadable")
    return `has an unreadable expiry time (${quoted(expiresAt)}), so it cannot be trusted`;
  if (e.state === "expired" && e.at.getTime() > now.getTime())
    return `is about to expire (at ${formatExpiry(expiresAt)})`;
  return `expired at ${formatExpiry(expiresAt)}`;
}
