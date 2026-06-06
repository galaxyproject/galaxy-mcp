export class GalaxyError extends Error {}

export class GalaxyAuthError extends GalaxyError {}
export class GalaxyNotFoundError extends GalaxyError {}

export class GalaxyConnectionError extends GalaxyError {
  constructor(
    message: string,
    readonly status?: number,
    readonly cause?: unknown,
  ) {
    super(message);
  }
}

/** tool_request.state === 'failed' -- the request couldn't even expand. */
export class ToolRequestRejectedError extends GalaxyError {
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
