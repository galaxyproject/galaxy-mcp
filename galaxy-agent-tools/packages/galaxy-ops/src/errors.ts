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
