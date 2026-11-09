import { createGalaxyContext, getUser } from "@galaxyproject/galaxy-ops";
import { abortReason } from "./abort";
import { describeNetworkError, markAnswered } from "./network";
import { AuthError, type BaseUrl, type WhoAmI } from "./types";
import { USER_AGENT } from "./user-agent";

type FetchImpl = typeof fetch;

/** For each request `whoami` makes: a Galaxy or proxy that accepts the connection and stalls must not hang a script. */
const REQUEST_TIMEOUT_MS = 15_000;

/**
 * Check, with no credentials at all, that `baseUrl` is a Galaxy's API before any token is
 * sent there. A Galaxy behind a `remote_user` proxy answers an HTML login page, and a wrong
 * URL often redirects (http to https, a missing path prefix): sending a key along would hand
 * it to whatever answers. Redirects are never followed; the `Location` is offered as a fix.
 *
 * `/api/version` is the question because Galaxy answers it to anyone, with JSON that carries
 * `version_major`. A 200 alone proves nothing: a login page is a 200 too.
 */
export async function probeGalaxy(
  baseUrl: BaseUrl,
  fetchImpl: FetchImpl = globalThis.fetch,
  signal?: AbortSignal,
  timeoutMs: number = REQUEST_TIMEOUT_MS,
): Promise<void> {
  const url = `${baseUrl}/api/version`;
  let res: Response;
  try {
    res = await fetchImpl(url, {
      redirect: "manual",
      headers: { accept: "application/json", "user-agent": USER_AGENT },
      signal: signal
        ? AbortSignal.any([signal, AbortSignal.timeout(timeoutMs)])
        : AbortSignal.timeout(timeoutMs),
    });
  } catch (e) {
    if (signal?.aborted) throw signal.reason ?? e;
    throw new AuthError(
      "network",
      describeNetworkError(e instanceof Error ? e : new Error(String(e)), baseUrl),
      { cause: e },
    );
  }
  if (res.status >= 300 && res.status < 400) {
    const where = res.headers.get("location");
    throw new AuthError(
      "protocol",
      `${baseUrl} redirects (HTTP ${res.status}) instead of answering, so no token was sent.` +
        (where ? ` Did you mean the address it points to: ${stripToApiBase(where, baseUrl)}?` : ""),
    );
  }
  // Both halves count: an error page can be JSON, and a login page can be a 200.
  let version: unknown;
  try {
    version = ((await res.json()) as { version_major?: unknown }).version_major;
  } catch {
    version = undefined;
  }
  if (!res.ok || typeof version !== "string") {
    throw new AuthError(
      "protocol",
      `${baseUrl} does not look like a Galaxy API (GET /api/version gave HTTP ${res.status} without a Galaxy version). ` +
        "Check the address, including any path prefix. No token was sent.",
    );
  }
}

/** A redirect target as a base URL suggestion: resolved against the request, `/api/version` removed. */
function stripToApiBase(location: string, baseUrl: string): string {
  try {
    const u = new URL(location, `${baseUrl}/api/version`);
    return `${u.origin}${u.pathname.replace(/\/api\/version\/?$/, "").replace(/\/+$/, "")}`;
  } catch {
    return "(unreadable)";
  }
}

/** One probe per Galaxy, shared by every `whoami` call that needs it; see `createWhoami`. */
interface SharedProbe {
  controller: AbortController;
  /** Calls still waiting on it. */
  waiters: number;
  settled: boolean;
  done: Promise<void>;
}

/** Settle with `p`, or reject as soon as `signal` aborts (what `p` is doing is left alone). */
function untilAborted<T>(p: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return p;
  if (signal.aborted) {
    p.catch(() => undefined);
    return Promise.reject(abortReason(signal));
  }
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => {
      reject(abortReason(signal));
    };
    signal.addEventListener("abort", onAbort, { once: true });
    p.then(resolve, reject).finally(() => {
      signal.removeEventListener("abort", onAbort);
    });
  });
}

const NOT_LOGGED_IN =
  "Galaxy did not accept the token: it may be invalid, replaced by a newer API key, expired, or belong to a deactivated account.";

/**
 * Ask Galaxy whose token this is, through galaxy-ops' `get_user` (`GET /api/users/current`,
 * which never follows a redirect). Galaxy 26.1 answers a key it does not know with 200 and an
 * anonymous record that has no `username`, so a 200 is the shape of the answer rather than
 * the answer: success means a real user, JSON with an `id` and a `username` or `email`.
 * The version probe runs first, once per Galaxy, so a token never reaches a non-Galaxy.
 *
 * The token goes out as `x-api-key`, as it does for every operation (galaxy-ops builds the
 * request), so what this accepts is what `gxy <operation>` will be able to use.
 */
export function createWhoami(
  fetchImpl: FetchImpl = globalThis.fetch,
  timeoutMs: number = REQUEST_TIMEOUT_MS,
) {
  const probed = new Map<string, SharedProbe>();
  return async function whoami(
    baseUrl: BaseUrl,
    token: string,
    signal?: AbortSignal,
  ): Promise<WhoAmI> {
    let probe = probed.get(baseUrl);
    if (!probe) {
      // The probe is shared, so one caller's cancel must not poison it for the next: it has its
      // own controller, and a caller that is cancelled just stops waiting. When the last waiter
      // has gone it is aborted, though, or a Galaxy that never answers would keep the process
      // alive after Ctrl-C until the probe's own timeout.
      const shared: SharedProbe = {
        controller: new AbortController(),
        waiters: 0,
        settled: false,
        done: Promise.resolve(),
      };
      shared.done = probeGalaxy(baseUrl, fetchImpl, shared.controller.signal, timeoutMs);
      probed.set(baseUrl, shared);
      shared.done.then(
        () => {
          shared.settled = true;
        },
        () => {
          shared.settled = true;
          if (probed.get(baseUrl) === shared) probed.delete(baseUrl);
        },
      );
      probe = shared;
    }
    const waiting = probe;
    waiting.waiters++;
    try {
      await untilAborted(waiting.done, signal);
    } finally {
      waiting.waiters--;
      if (waiting.waiters === 0 && !waiting.settled) {
        waiting.controller.abort();
        if (probed.get(baseUrl) === waiting) probed.delete(baseUrl);
      }
    }
    signal?.throwIfAborted();

    const ctx = createGalaxyContext({
      baseUrl,
      apiKey: token,
      fetchImpl: (input, init) => {
        // galaxy-ops knows nothing of our signal, timeout or User-Agent, so all three go on here.
        const timeout = AbortSignal.timeout(timeoutMs);
        const req = new Request(input, {
          ...init,
          signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
        });
        req.headers.set("user-agent", USER_AGENT);
        return fetchImpl(req);
      },
    });
    let user: Awaited<ReturnType<typeof getUser>>;
    try {
      user = await getUser({}, ctx);
    } catch (e) {
      if (signal?.aborted) throw signal.reason ?? e;
      throw mapFailure(e, baseUrl);
    }
    const u = user as { id?: unknown; username?: unknown; email?: unknown; is_admin?: unknown };
    const name =
      typeof u.username === "string" && u.username
        ? u.username
        : typeof u.email === "string"
          ? u.email
          : "";
    if (typeof u.id !== "string" || !u.id || !name)
      throw new AuthError("not_logged_in", NOT_LOGGED_IN);
    return {
      username: name,
      ...(typeof u.email === "string" && u.email ? { email: u.email } : {}),
      isAdmin: u.is_admin === true,
    };
  };
}

function mapFailure(e: unknown, baseUrl: string): AuthError {
  const err = e as { kind?: string; status?: number; http?: { status: number | null } };
  const status = err.status ?? err.http?.status;
  if ((err.kind === "connection" && status === undefined) || status === null) {
    // Names NODE_EXTRA_CA_CERTS / NODE_USE_ENV_PROXY, as the OAuth calls do.
    return new AuthError("network", describeNetworkError(e as Error, baseUrl), { cause: e });
  }
  if (typeof status === "number") {
    // The server answered, so these are `network` (retry later) without being a transport failure:
    // `markAnswered` keeps the proxy hint off them.
    if (status >= 500)
      return markAnswered(
        new AuthError("network", `${baseUrl} answered HTTP ${status}.`, { cause: e }),
      );
    // Not a refusal of the token: say what it is, or the person re-creates a good key for nothing.
    if (status === 429)
      return markAnswered(
        new AuthError(
          "network",
          `${baseUrl} is rate-limiting requests (HTTP 429). Wait a minute and try again.`,
          { cause: e },
        ),
      );
    // galaxy-ops refuses to follow a redirect with a key on the request, so the key went nowhere
    // else. Not a refusal of the token either: the address is wrong, as the probe says of a redirect.
    if (status >= 300 && status < 400)
      return new AuthError(
        "protocol",
        `${baseUrl} redirects (HTTP ${status}) the user lookup instead of answering, so the token was not sent on. Check the address, including any path prefix.`,
        { cause: e },
      );
    if (status === 404) {
      return new AuthError(
        "protocol",
        `${baseUrl} answered HTTP 404 for the user lookup, so this URL does not look like a Galaxy API. Check the address, including any path prefix.`,
        { cause: e },
      );
    }
  }
  // Auth refusals, HTML or non-JSON bodies, anything else that is not a user record.
  return new AuthError("not_logged_in", NOT_LOGGED_IN, { cause: e });
}
