import * as client from "openid-client";
import { isLoopback } from "./base-url";
import { describeNetworkError, isTlsError } from "./network";
import { sanitize } from "./sanitize";
import { isPlainToken } from "./token-shape";
import {
  AuthError,
  CLIENT_ID,
  type AuthServer,
  type BaseUrl,
  type PendingDeviceLogin,
  type TokenGrant,
} from "./types";
import { abortReason } from "./abort";
import { trustedVerificationUri } from "./verification-uri";

export interface DiscoverOptions {
  /** OAuth client id to present. Default `CLIENT_ID`; callers feed `GXY_OAUTH_CLIENT_ID` here. */
  clientId?: string;
  /** Replaces `fetch` for every request (tests, proxies). */
  fetch?: typeof fetch;
  /**
   * Cancels discovery. Not the later calls: `startDeviceLogin`, `waitForApproval` and `revoke`
   * each take their own, because the signal that cancels a login must not reach the revocation of
   * a token that login minted a moment before (it would be aborted before it started).
   */
  signal?: AbortSignal;
  /**
   * The least time between two polls, whatever interval the server asks for. Default 1000: a
   * server that asks for `interval: 0.001` would otherwise be polled a thousand times a second.
   * Tests with a fast fake Galaxy lower it.
   */
  minIntervalMs?: number;
}

/**
 * Bounds on numbers the server chooses. `setTimeout` fires after 1 ms for anything above 2^31 ms,
 * so a huge `expires_in`, `interval` or `Retry-After` would end a wait at once or make the poll a
 * hot loop; and device codes live minutes, not hours.
 */
const MAX_EXPIRES_IN_S = 3600;
const MAX_INTERVAL_S = 60;
const MAX_RETRY_AFTER_S = 60;
const DEFAULT_MIN_INTERVAL_MS = 1000;

/** The `grant_type` of RFC 8628 §3.4, sent by `genericGrantRequest` because the polling loop is ours. */
const DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code";

/** The endpoints that carry a token or a device code, and so must be on the issuer's own origin. */
const ENDPOINTS_ON_ISSUER_ORIGIN = [
  "device_authorization_endpoint",
  "token_endpoint",
  "revocation_endpoint",
] as const;

/**
 * `fetch` with two guards of our own. Plain http goes only to a loopback address, however
 * the library was configured (its `allowInsecureRequests` is all-or-nothing), because over
 * plain http to anything else the device code and then the token would cross the network in
 * the clear. And the caller's signal cancels every request, so a Ctrl-C ends a request that
 * is in flight rather than only the wait between two.
 */
function guardedFetch(base: typeof fetch, signal: AbortSignal | undefined): typeof fetch {
  return (input, init) => {
    const target = new URL(
      typeof input === "string" ? input : input instanceof URL ? input.href : input.url,
    );
    if (target.protocol === "http:" && !isLoopback(target.hostname)) {
      return Promise.reject(
        new AuthError(
          "insecure",
          `Refusing plain http to ${target.host}: a token would cross the network unencrypted.`,
        ),
      );
    }
    const combined =
      signal && init?.signal
        ? AbortSignal.any([signal, init.signal])
        : (signal ?? init?.signal ?? undefined);
    return base(input, { ...init, signal: combined });
  };
}

/**
 * Transient failures worth another poll: no answer, a timeout, a 5xx or a 429. Never TLS
 * trouble, which would only fail the same way until the code ran out.
 *
 * Returns the least the server wants us to wait in milliseconds (`Retry-After`, or 0), or
 * `undefined` when the failure is not one to retry.
 */
function transientDelayMs(err: unknown): number | undefined {
  // RFC 9110 §10.2.3: seconds, or an HTTP date; either way capped (see MAX_RETRY_AFTER_S).
  const retryAfter = (response: Response | undefined): number => {
    const h = response?.headers.get("retry-after")?.trim();
    if (!h) return 0;
    const n = Number(h);
    const seconds = Number.isFinite(n) ? n : (Date.parse(h) - Date.now()) / 1000;
    return Number.isFinite(seconds) && seconds > 0
      ? Math.min(seconds, MAX_RETRY_AFTER_S) * 1000
      : 0;
  };
  const retryableStatus = (r: Response | undefined) =>
    r !== undefined && (r.status >= 500 || r.status === 429);
  if (err instanceof client.ResponseBodyError)
    return retryableStatus(err.response) ? retryAfter(err.response) : undefined;
  if (err instanceof client.ClientError) {
    const response = err.cause instanceof Response ? err.cause : undefined;
    return retryableStatus(response) ? retryAfter(response) : undefined;
  }
  if (err instanceof TypeError) {
    // A coded TypeError is the library refusing our own arguments or configuration, which a
    // retry cannot fix; the plain "fetch failed" is the network.
    const code = (err as { code?: unknown }).code;
    if (typeof code === "string" && /^(?:ERR_|OAUTH_)/.test(code)) return undefined;
    return isTlsError(err) ? undefined : 0;
  }
  if (err instanceof Error && err.name === "TimeoutError") return 0;
  return undefined;
}

/** Sleep that ends early, rejecting, when the signal aborts, so Ctrl-C never waits out an interval. */
function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(abortReason(signal));
      return;
    }
    const done = () => {
      clearTimeout(timer);
      signal.removeEventListener("abort", onAbort);
    };
    const onAbort = () => {
      done();
      reject(abortReason(signal));
    };
    const timer = setTimeout(() => {
      done();
      resolve();
    }, ms);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

/**
 * RFC 8414 discovery against the Galaxy at `baseUrl` (its issuer), returning the device flow
 * and revocation as an `AuthServer`.
 *
 * The client is public: it holds no secret and says so (`None()`), so there is nothing to leak
 * to a server that asks. Plain http is allowed only to a loopback host, for a Galaxy on this
 * machine; `normalizeBaseUrl` already refuses it elsewhere, and this checks again because a
 * token is what is at stake. A Galaxy under a path prefix is found by RFC 8414's path
 * insertion, which the library does, and the metadata's `issuer` must equal `baseUrl` or the
 * metadata is refused as being about another server.
 *
 * No released Galaxy serves the device flow yet (the server side is galaxyproject/galaxy#23645):
 * Galaxy 26.1 answers the well-known URL with an HTML 404 page, which comes out of here as
 * `unsupported`, which a caller can turn into a hint to use `--with-token`.
 *
 * Every failure is an `AuthError`; see `types.ts` for what each kind means.
 */
export async function discover(
  baseUrl: BaseUrl,
  options: DiscoverOptions = {},
): Promise<AuthServer> {
  const url = new URL(baseUrl);
  const loopback = isLoopback(url.hostname);
  if (url.protocol === "http:" && !loopback) {
    throw new AuthError(
      "insecure",
      `Refusing plain http to ${url.host}: a token would cross the network unencrypted.`,
    );
  }
  const clientId = options.clientId || CLIENT_ID;

  const baseFetch = options.fetch ?? globalThis.fetch;
  let config: client.Configuration;
  try {
    config = await client.discovery(url, clientId, undefined, client.None(), {
      algorithm: "oauth2",
      [client.customFetch]: guardedFetch(baseFetch, options.signal),
      // The library applies this to every request; `guardedFetch` narrows it to loopback URLs.
      // eslint-disable-next-line @typescript-eslint/no-deprecated -- deprecated only to make a security-sensitive option stand out, and here it is gated on loopback
      ...(loopback ? { execute: [client.allowInsecureRequests] } : {}),
    });
  } catch (err) {
    if (options.signal?.aborted) throw options.signal.reason ?? err;
    throw mapError(err, "discovery", baseUrl);
  }

  const meta = config.serverMetadata();
  // Logout sends the token to `revocation_endpoint` and polling sends the device code to
  // `token_endpoint`: none may point anywhere but the Galaxy that was asked, or the metadata
  // could send the device code and the token to another host.
  for (const name of ENDPOINTS_ON_ISSUER_ORIGIN) {
    const value = meta[name];
    if (typeof value !== "string") continue;
    let origin: string | undefined;
    try {
      origin = new URL(value).origin;
    } catch {
      /* falls through to the refusal */
    }
    if (origin !== url.origin) {
      throw new AuthError(
        "insecure",
        `${baseUrl} publishes its ${name} on another origin (${sanitize(origin ?? "not a URL")}), so no token or code was sent.`,
      );
    }
  }
  const supportsDeviceFlow = typeof meta.device_authorization_endpoint === "string";
  // A device endpoint with nowhere to poll would leave a code on the server and fail the first
  // poll, an interval later: refuse it here, before anything is sent.
  if (supportsDeviceFlow && typeof meta.token_endpoint !== "string") {
    throw new AuthError(
      "protocol",
      `${baseUrl} offers device login but publishes no token_endpoint to collect the token from.`,
    );
  }
  const supportsRevocation = typeof meta.revocation_endpoint === "string";
  const minIntervalMs = options.minIntervalMs ?? DEFAULT_MIN_INTERVAL_MS;

  /**
   * A configuration whose requests are cancelled by `signal`. One per call, because
   * openid-client's functions take no signal: the only way into a request is the `fetch` the
   * configuration holds, and sharing one would let a Ctrl-C meant for the wait reach a revocation.
   */
  const scoped = (signal: AbortSignal | undefined): client.Configuration => {
    const c = new client.Configuration(meta, clientId, undefined, client.None());
    c[client.customFetch] = guardedFetch(baseFetch, signal);
    // eslint-disable-next-line @typescript-eslint/no-deprecated -- as above: gated on loopback, and `guardedFetch` narrows it
    if (loopback) client.allowInsecureRequests(c);
    return c;
  };

  return {
    issuer: meta.issuer,
    supportsDeviceFlow,
    supportsRevocation,

    async startDeviceLogin(scopes, startSignal) {
      if (!supportsDeviceFlow) {
        throw new AuthError("unsupported", `${baseUrl} does not offer the device login flow.`);
      }
      // A cancel before this must not leave a code waiting on the server.
      startSignal?.throwIfAborted();
      const requested = scopes ?? [];
      let response: Awaited<ReturnType<typeof client.initiateDeviceAuthorization>>;
      try {
        response = await client.initiateDeviceAuthorization(
          scoped(startSignal),
          requested.length ? { scope: requested.join(" ") } : {},
        );
      } catch (err) {
        if (startSignal?.aborted) throw startSignal.reason ?? err;
        throw mapError(err, "device", baseUrl);
      }
      // The page must be where the person can safely be sent, whoever receives this. The code
      // already exists on the server, so this ends the login rather than hiding the page.
      const verificationUri = trustedVerificationUri(response.verification_uri, baseUrl);
      if (verificationUri === undefined) {
        throw new AuthError(
          "insecure",
          `${baseUrl} asked for the code to be entered at ${sanitize(response.verification_uri)}, which is not an https page on its own host, so the login was stopped.`,
        );
      }
      const expiresAt = new Date(
        Date.now() + Math.min(Math.max(response.expires_in, 1), MAX_EXPIRES_IN_S) * 1000,
      );

      // Only the user code and `verification_uri` leave this function. The device code stays in
      // this closure, so nothing downstream can print it: it is the secret half of the pair
      // (RFC 8628 §3.2), and whoever holds it collects the token once the person approves.
      // `verification_uri_complete` is ignored even when sent: the code is shown and typed, not
      // carried in a link, so an approval means the person saw it where the command ran (§5.4 is
      // the remote-phishing risk this narrows).
      const pending: PendingDeviceLogin = {
        userCode: response.user_code,
        verificationUri,
        expiresAt,
        async waitForApproval(signal) {
          // openid-client's default deadline is expires_in, but giving it a signal replaces
          // that, so the deadline is ours: either the caller aborts or the code expires.
          // A plain timer, cleared below, rather than `AbortSignal.timeout`: tests can fake it.
          const deadlineControl = new AbortController();
          const deadlineTimer = setTimeout(
            () => {
              deadlineControl.abort();
            },
            Math.max(0, expiresAt.getTime() - Date.now()),
          );
          const deadline = deadlineControl.signal;
          const combined = signal ? AbortSignal.any([signal, deadline]) : deadline;
          // Each poll is made with `combined`, so a Ctrl-C or the code's expiry ends a request in
          // flight rather than only the wait between two (the library's own 30 s timeout would
          // otherwise be the limit).
          const poll = scoped(combined);
          // RFC 8628 §3.5: poll every `interval` seconds (default 5), and 5 more for this and every
          // later poll after a `slow_down`. The loop is ours, not openid-client's, because its
          // waits ignore the signal for up to 5 s and its timers keep the process alive after a
          // Ctrl-C.
          let intervalMs = Math.max(
            minIntervalMs,
            Math.min(response.interval ?? 5, MAX_INTERVAL_S) * 1000,
          );
          let failures = 0;
          let lastFailure = "";
          let delayMs = intervalMs;
          const expired = (cause: unknown) =>
            new AuthError(
              "expired",
              "The login code expired before it was approved." +
                // An outage that outlasts the code looks the same as nobody approving it, so say
                // what the polls got.
                (failures > 0
                  ? ` The last ${failures} poll${failures === 1 ? "" : "s"} to ${baseUrl} failed (${lastFailure}).`
                  : ""),
              { cause },
            );
          try {
            for (;;) {
              try {
                await sleep(delayMs, combined);
              } catch (e) {
                if (signal?.aborted) throw signal.reason ?? e;
                throw expired(e);
              }
              try {
                const tokens = await client.genericGrantRequest(poll, DEVICE_GRANT, {
                  device_code: response.device_code,
                });
                return toGrant(tokens, requested);
              } catch (err) {
                if (signal?.aborted) throw signal.reason ?? err;
                if (deadline.aborted) throw expired(err);
                if (
                  err instanceof client.ResponseBodyError &&
                  (err.error === "authorization_pending" || err.error === "slow_down")
                ) {
                  if (err.error === "slow_down") intervalMs += 5000;
                  failures = 0;
                  delayMs = intervalMs;
                  continue;
                }
                // Anything else ends the wait. RFC 8628 §3.5 says `access_denied` and `expired_token`
                // do, and an error we do not recognise is not one to keep asking the server about.
                const transient = transientDelayMs(err);
                if (transient === undefined) throw mapError(err, "poll", baseUrl);
                failures += 1;
                lastFailure = briefReason(err);
                // Modest exponential backoff, and never sooner than the server asked.
                delayMs = Math.max(
                  intervalMs,
                  transient,
                  Math.min(60_000, intervalMs * 2 ** failures),
                );
              }
            }
          } finally {
            clearTimeout(deadlineTimer);
          }
        },
      };
      return pending;
    },

    async revoke(token, signal) {
      if (!supportsRevocation) {
        throw new AuthError("unsupported", `${baseUrl} does not publish a revocation endpoint.`);
      }
      try {
        // RFC 7009 §2.1: the hint saves the server working out which kind of token this is. A
        // token that is already dead still gets a 200 (§2.2), so a late logout is not an error.
        await client.tokenRevocation(scoped(signal), token, { token_type_hint: "access_token" });
      } catch (err) {
        throw mapError(err, "revoke", baseUrl);
      }
    },
  };
}

/** What a failed poll got, in a few words for the message that says why a code ran out. */
function briefReason(err: unknown): string {
  const response =
    err instanceof client.ResponseBodyError
      ? err.response
      : err instanceof client.ClientError && err.cause instanceof Response
        ? err.cause
        : undefined;
  if (response) return `HTTP ${response.status}`;
  if (err instanceof Error)
    return sanitize(err.cause instanceof Error ? err.cause.message : err.message);
  return "unknown error";
}

function toGrant(tokens: client.TokenEndpointResponse, requested: string[]): TokenGrant {
  if (typeof tokens.access_token !== "string" || !tokens.access_token) {
    throw new AuthError("protocol", "The token response had no access token.");
  }
  if (!isPlainToken(tokens.access_token)) {
    // Never echo it: it is a secret, and it is not a token we can use.
    throw new AuthError(
      "protocol",
      "The token response carried an access token with whitespace or non-ASCII characters, so it was not used.",
    );
  }
  // RFC 6749 §5.1: `scope` is omitted when it equals what was requested.
  const scopes =
    typeof tokens.scope === "string" ? tokens.scope.split(/\s+/).filter(Boolean) : [...requested];
  return {
    accessToken: tokens.access_token,
    ...(typeof tokens.expires_in === "number" ? { expiresIn: tokens.expires_in } : {}),
    scopes,
  };
}

type Phase = "discovery" | "device" | "poll" | "revoke";

/**
 * What a bad answer to the well-known URL means. Only "there is no such document" is
 * `unsupported`, which a caller can turn into the hint to paste a key: a 404 (what Galaxy 26.1 sends)
 * or a 200 that is not JSON (a web page standing in for it). A proxy that is briefly down
 * (5xx, 429) is not a Galaxy without the feature, so it is a network failure to retry; a redirect
 * is somewhere else's answer and never followed with anything attached, so it is a protocol
 * matter and names where it pointed.
 */
function mapDiscoveryResponse(err: client.ClientError, baseUrl: string, detail: string): AuthError {
  const response = err.cause instanceof Response ? err.cause : undefined;
  if (
    err.code === "OAUTH_RESPONSE_IS_NOT_JSON" ||
    response?.status === 404 ||
    response?.status === 410
  ) {
    return new AuthError("unsupported", `${baseUrl} publishes no OAuth metadata (${detail}).`, {
      cause: err,
    });
  }
  if (!response)
    return new AuthError("protocol", `${baseUrl} answered unexpectedly (${detail}).`, {
      cause: err,
    });
  const { status } = response;
  if (status >= 500 || status === 429) {
    return new AuthError(
      "network",
      `${baseUrl} answered ${status} when asked for its OAuth metadata. It may be restarting; try again in a moment.`,
      { cause: err },
    );
  }
  if (status >= 300 && status < 400) {
    const location = response.headers.get("location");
    let where = "somewhere else";
    try {
      if (location) where = new URL(location, baseUrl).host;
    } catch {
      /* keeps "somewhere else" */
    }
    return new AuthError(
      "protocol",
      `${baseUrl} redirected (${status}) its OAuth metadata to ${sanitize(where)} instead of serving it. Check the Galaxy URL (http versus https, or a missing path prefix).`,
      { cause: err },
    );
  }
  return new AuthError(
    "protocol",
    `${baseUrl} answered ${status} when asked for its OAuth metadata.`,
    {
      cause: err,
    },
  );
}

/** Turn whatever openid-client threw into an `AuthError` of the right kind. */
function mapError(err: unknown, phase: Phase, baseUrl: string): AuthError {
  if (err instanceof AuthError) return err;
  // Server text reaches these messages (error texts, response excerpts), and they get printed.
  const detail = sanitize(err instanceof Error ? err.message : String(err));

  if (err instanceof client.ResponseBodyError) {
    switch (err.error) {
      case "access_denied":
        return new AuthError("denied", "The login was denied on the approval page.", {
          cause: err,
        });
      case "expired_token":
        return new AuthError("expired", "The login code expired before it was approved.", {
          cause: err,
        });
      default:
        return new AuthError(
          "protocol",
          `${baseUrl} refused the request: ${sanitize(err.error)}${err.error_description ? ` (${sanitize(err.error_description)})` : ""}.`,
          { cause: err },
        );
    }
  }

  if (err instanceof client.ClientError) {
    switch (err.code) {
      case "OAUTH_JSON_ATTRIBUTE_COMPARISON_FAILED":
        return new AuthError(
          "insecure",
          `${baseUrl} publishes OAuth metadata for a different issuer, so it cannot be trusted.`,
          { cause: err },
        );
      case "OAUTH_HTTP_REQUEST_FORBIDDEN":
      case "OAUTH_REQUEST_PROTOCOL_FORBIDDEN":
        return new AuthError("insecure", `Refusing plain http to ${baseUrl}.`, { cause: err });
      case "OAUTH_RESPONSE_IS_NOT_CONFORM":
      case "OAUTH_RESPONSE_IS_NOT_JSON":
        if (phase === "discovery") return mapDiscoveryResponse(err, baseUrl, detail);
        return new AuthError("protocol", `${baseUrl} answered unexpectedly (${detail}).`, {
          cause: err,
        });
      case undefined:
      default:
        return new AuthError(
          "protocol",
          `${baseUrl} sent a response that does not follow the OAuth specs (${detail}).`,
          { cause: err },
        );
    }
  }

  // fetch failures (refused, DNS, reset, timeout) surface as TypeError or a timeout DOMException.
  if (
    err instanceof TypeError ||
    (err instanceof Error && (err.name === "TimeoutError" || err.name === "AbortError"))
  ) {
    return new AuthError("network", describeNetworkError(err, baseUrl), { cause: err });
  }
  return new AuthError("protocol", `Unexpected failure talking to ${baseUrl}: ${detail}.`, {
    cause: err,
  });
}
