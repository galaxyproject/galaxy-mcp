import { httpFailureFacts } from "./http-failure";
import type { HttpFailureFacts } from "./python-failure";

export type GalaxyErrorKind =
  | "auth"
  | "not_found"
  | "validation"
  | "connection"
  | "version"
  | "tool_request_rejected"
  | "job_failed"
  | "out_of_scope"
  | "handle_kind_mismatch"
  | "unknown";

export class GalaxyError extends Error {
  readonly kind: GalaxyErrorKind = "unknown";
  /**
   * What the request that failed left behind, when this error came from one.
   *
   * Present on an HTTP failure and absent on a refusal of our own, and the boundary reads
   * it exactly that way: a failure carrying facts is worded the way the Python server words
   * one, from the operation's own action and context, and one carrying none already says
   * everything it has to say. `message` is this library's own either way -- a library caller
   * reads the same short sentence it always did.
   */
  http?: HttpFailureFacts;
  /** Galaxy's own `err_code` (e.g. 403004), when the reply carried one. */
  errCode?: number;
}

export class GalaxyAuthError extends GalaxyError {
  readonly kind = "auth" as const;
}
export class GalaxyNotFoundError extends GalaxyError {
  readonly kind = "not_found" as const;
}

/**
 * The caller has to change its input; retrying the same call cannot help.
 *
 * Distinct from `connection` on purpose: an agent that reads "connection" backs
 * off and retries, which for a rejected argument loops forever, and the CLI
 * exits 69 "service unavailable" for what is a usage error.
 */
export class GalaxyValidationError extends GalaxyError {
  readonly kind = "validation" as const;
}

export class GalaxyConnectionError extends GalaxyError {
  readonly kind = "connection" as const;
  constructor(
    message: string,
    readonly status?: number,
    readonly cause?: unknown,
  ) {
    super(message);
  }
}

/**
 * The connected Galaxy is older than the operation needs, and nothing was sent.
 *
 * Its own kind because none of the others tells the truth here: the call did not fail, it
 * was never made, and neither a retry nor a different argument can change the answer -- the
 * server is what has to change.
 */
export class GalaxyVersionError extends GalaxyError {
  readonly kind = "version" as const;
}

/**
 * The call is outside what the host allowed this run: a history it may not touch, an
 * operation it may not call, or a record that turned out to belong to another history.
 *
 * Its own kind because none of the others tells the truth here either: the argument may be a
 * perfectly good id, the server would have answered, and the host is the one who decides --
 * so a retry cannot help and neither can a different spelling. Refused before anything is sent
 * where the argument says enough, and after the read where only the record could say.
 */
export class GalaxyScopeError extends GalaxyError {
  readonly kind = "out_of_scope" as const;
}

/**
 * An id passed as one kind of thing after an earlier call in this run returned it as another:
 * a job id where a dataset id belongs. Galaxy's ids carry no type, so the only thing that can
 * say is the run's own memory, and the caller has to pick the id it meant.
 */
export class GalaxyHandleKindError extends GalaxyError {
  readonly kind = "handle_kind_mismatch" as const;
}

/** tool_request.state === 'failed' -- the request couldn't even expand. */
export class ToolRequestRejectedError extends GalaxyError {
  readonly kind = "tool_request_rejected" as const;
  constructor(
    readonly toolId: string,
    readonly errMsg: string,
    readonly toolRequestId?: string,
  ) {
    super(`Tool request for ${toolId} was rejected: ${errMsg}`);
  }
}

/** A spawned job reached a terminal-failure state -- it ran and failed. */
export class JobFailedError extends GalaxyError {
  readonly kind = "job_failed" as const;
  constructor(
    readonly jobId: string,
    readonly state: string,
    readonly stderr?: string,
  ) {
    super(`Job ${jobId} failed (state=${state})`);
  }
}

/**
 * Classify a failed request AND carry what it needs to be worded the other server's way.
 *
 * The classification is `classifyHttp`'s and has not moved: it is what `errorKind` and the
 * CLI's exit code are read off, and it stays the status and nothing else. What is added is
 * the reply itself -- status, method, URL, bytes -- taken from the middleware in client.ts,
 * because the sentence quotes all four and none of them survives the parse.
 */
export function httpError(
  response: { status: number } | undefined,
  errorBody: unknown,
): GalaxyError {
  const status = response?.status ?? 0;
  const error = classifyHttp(status, errorBody);
  const facts = httpFailureFacts(response);
  error.http = facts ?? {
    status,
    method: "GET",
    url: "",
    bodyText: typeof errorBody === "string" ? errorBody : JSON.stringify(errorBody ?? null),
  };
  return error;
}

/** Galaxy's own `err_msg` and `err_code` off an error body, when it sent them. */
function galaxyErrorFields(errorBody: unknown): { errMsg?: string; errCode?: number } {
  if (!errorBody || typeof errorBody !== "object") return {};
  const body = errorBody as { err_msg?: unknown; err_code?: unknown };
  const errMsg = body.err_msg == null || body.err_msg === "" ? undefined : String(body.err_msg);
  const errCode = typeof body.err_code === "number" ? body.err_code : undefined;
  return { errMsg, errCode };
}

/**
 * Classify an openapi-fetch failure on the HTTP status only -- never substring scans.
 *
 * The status picks the kind; Galaxy's err_msg, when there is one, is the text. A 403 is not
 * always a bad key -- Galaxy refuses a URL its fetch allowlist does not name with a 403 too --
 * so the generic auth sentence is only the fallback for a reply that says nothing itself.
 */
export function classifyHttp(status: number, errorBody: unknown): GalaxyError {
  const { errMsg, errCode } = galaxyErrorFields(errorBody);
  let error: GalaxyError;
  if (status === 401 || status === 403) error = new GalaxyAuthError(errMsg ?? `Unauthorized (${status})`);
  else if (status === 404) error = new GalaxyNotFoundError(errMsg ?? "Not found (404)");
  else error = new GalaxyConnectionError(errMsg ?? `HTTP ${status}`, status);
  if (errCode !== undefined) error.errCode = errCode;
  return error;
}

/**
 * What an unfollowed redirect says, on every surface.
 *
 * Its own text rather than the library's quote of the reply: a proxy's 301 body is HTML, and
 * the Location -- the one fact that tells the caller what to fix -- is not in it.
 */
/** 304 is a cache answer to a conditional request, not a redirect: it stays a plain non-2xx. */
export const isRedirectStatus = (status: number): boolean =>
  status >= 300 && status < 400 && status !== 304;

const REDIRECT_TARGET_MAX = 200;

/**
 * Where a redirect pointed, fit to show a user: resolved against the request, without
 * userinfo, query or fragment (a presigned signature has no business in an error), and capped.
 */
export function redirectTarget(location: string, from: string): string {
  let shown: string;
  try {
    const u = new URL(location, from);
    u.username = "";
    u.password = "";
    u.search = "";
    u.hash = "";
    shown = u.href;
  } catch {
    shown = location.split(/[?#]/)[0]!;
  }
  return shown.length > REDIRECT_TARGET_MAX ? `${shown.slice(0, REDIRECT_TARGET_MAX)}...` : shown;
}

export function redirectRefusal(status: number | null, target: string | undefined): string {
  const what = status === null ? "Galaxy answered with a redirect" : `Galaxy answered ${status}`;
  const where = target ? `; Galaxy redirected to ${target}` : "";
  return `${what} and redirects are not followed${where} -- check GALAXY_URL`;
}

/**
 * The error for a request that never completed: a refused connection, a DNS failure, an
 * abort. There is no status and no body, and what the runtime said for itself is the whole
 * text -- which is the shape the other server's failure is in too, and not the same words.
 * Its client's message is requests', ours is the runtime's, and no fixture pins either.
 */
export function transportFailure(request: { method: string; url: string }, error: unknown): Error {
  const said = error instanceof Error ? error.message : String(error);
  const failure = new GalaxyConnectionError(said, undefined, error);
  failure.http = {
    status: null,
    method: request.method.toUpperCase(),
    url: request.url,
    bodyText: "",
    transportMessage: said,
  };
  return failure;
}
