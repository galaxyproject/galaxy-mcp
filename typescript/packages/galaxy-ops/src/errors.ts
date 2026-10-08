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

/** Classify an openapi-fetch failure on the HTTP status only -- never substring scans. */
export function classifyHttp(status: number, errorBody: unknown): GalaxyError {
  if (status === 401 || status === 403) return new GalaxyAuthError(`Unauthorized (${status})`);
  if (status === 404) return new GalaxyNotFoundError("Not found (404)");
  const msg =
    errorBody && typeof errorBody === "object" && "err_msg" in errorBody
      ? String((errorBody as { err_msg: unknown }).err_msg)
      : `HTTP ${status}`;
  return new GalaxyConnectionError(msg, status);
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
