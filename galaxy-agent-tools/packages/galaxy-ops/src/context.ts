import { createGalaxyClient, type GalaxyClient } from "./client";
import { classifyHttp, GalaxyConnectionError, type GalaxyError } from "./errors";
import { parseGalaxyVersion, type GalaxyVersion } from "./version";

export interface PollPolicy {
  intervalMs: number;
  maxIntervalMs: number;
  backoff: number; // multiplier per attempt
  jitter: number; // 0..1 fraction of interval added randomly
  timeoutMs: number;
}

export const DEFAULT_POLL: PollPolicy = {
  intervalMs: 1000,
  maxIntervalMs: 10_000,
  backoff: 1.5,
  jitter: 0.1,
  timeoutMs: 600_000,
};

/** Where a context's version came from. "supplied" means the caller told us and we never asked. */
export type GalaxyVersionSource = "supplied" | "server" | "unknown";

/**
 * Everything one /api/version lookup produced, so that the guard and anything reporting the
 * server to a caller work from the same answer instead of probing separately and disagreeing.
 */
export interface GalaxyVersionReport {
  /** version_major, parsed. Absent when the server would not say or said something unreadable. */
  readonly version?: GalaxyVersion;
  /** The payload as sent, for a caller that wants the whole thing rather than the comparison. */
  readonly payload?: Record<string, unknown>;
  /** Why it came back empty, for a caller that has to report the failure rather than shrug. */
  readonly error?: GalaxyError;
  /** A supplied version is still a known one -- a reporting caller must not mistake it for none. */
  readonly source: GalaxyVersionSource;
}

/**
 * Where one call's ops leave facts its envelope needs and `run()` does not return.
 *
 * Written through the typed helpers in operations/envelope-facts.ts and read by
 * the projection of the same call. Absent on a context a library caller built,
 * which is what makes recording a no-op for them.
 */
export type EnvelopeFacts = Map<symbol, unknown>;

export interface GalaxyContext {
  readonly client: GalaxyClient;
  readonly baseUrl?: string;
  readonly apiKey?: string;
  readonly serverVersion?: string;
  /**
   * The connected server's version, asked for once and remembered. The single source of truth:
   * no op fetches /api/version for itself, or it can reach a different conclusion than the
   * guard. Optional because a hand-assembled context has none, and a context that cannot say is
   * a server whose version is unknown -- which refuses nothing.
   */
  readonly galaxyVersion?: () => Promise<GalaxyVersionReport>;
  readonly poll: PollPolicy;
  readonly signal?: AbortSignal;
  /**
   * This call's envelope-fact collector, put here by `runWithEnvelope` and by
   * nothing else. One per call, never shared: two calls in flight at once must
   * not be able to read each other's numbers.
   */
  readonly envelopeFacts?: EnvelopeFacts;
}

export interface CreateContextOptions {
  baseUrl: string;
  apiKey: string;
  serverVersion?: string;
  poll?: Partial<PollPolicy>;
  signal?: AbortSignal;
  fetchImpl?: typeof fetch; // tests only
  versionProbeTimeoutMs?: number; // tests only
}

/**
 * How long the version probe waits before giving up on the server.
 *
 * /api/version is a tiny unauthenticated GET that a healthy Galaxy answers in well under a
 * second, so a few seconds is generous for a slow link while still short enough that a hung
 * version route cannot hold up an endpoint that works perfectly well. Giving up leaves the
 * version unknown, and unknown refuses nothing.
 */
export const VERSION_PROBE_TIMEOUT_MS = 3_000;

/**
 * A promise that rejects when a signal aborts, and otherwise never settles.
 *
 * Raced against the request below, rather than trusting the request to come back. The signal
 * cancels the fetch, which is what frees the socket, but whether that cancellation arrives as a
 * rejected promise is the fetch implementation's business -- and it is the probe settling that
 * every gated op is waiting on. So giving up is decided here, where the deadline is, and does
 * not depend on anything downstream noticing it.
 */
function rejectWhenAborted(signal: AbortSignal): Promise<never> {
  return new Promise((_resolve, reject) => {
    if (signal.aborted) return reject(signal.reason);
    signal.addEventListener("abort", () => reject(signal.reason), { once: true });
  });
}

/**
 * Ask the server what version it is and hand every later caller the same answer.
 *
 * Lazy rather than eager because createGalaxyContext is synchronous and because a context
 * that is built and never used -- the CLI builds one before it knows the arguments parse --
 * should not cost a request. It is the promise that is memoized, not the value, so ops
 * running concurrently share one lookup.
 *
 * An answer is kept for the life of the context; a failure is dropped as soon as it settles, so
 * the next call asks again. One context can outlive the outage that caused a failure -- the MCP
 * server builds a single context for the whole process -- and one 502 from a restarting Galaxy
 * must not leave every requirement unchecked until someone restarts it. Nothing here tries to
 * keep two callers consistent with each other: a call that must not change its mind half way
 * pins one report onto the context it passes down (see guardVersion).
 *
 * A supplied version that does not parse falls through to the network rather than standing in
 * as "unknown" for good, so a blank value from a config file or a shell variable cannot quietly
 * switch the checking off.
 */
function versionLookup(
  client: GalaxyClient,
  supplied?: string,
  callerSignal?: AbortSignal,
  timeoutMs: number = VERSION_PROBE_TIMEOUT_MS,
): () => Promise<GalaxyVersionReport> {
  const said = parseGalaxyVersion(supplied);
  let pending: Promise<GalaxyVersionReport> | undefined;
  return () => {
    // A supplied version answers for the server, so there is nothing to fetch and no payload.
    if (said) return Promise.resolve({ version: said, source: "supplied" as const });
    pending ??= (async (): Promise<GalaxyVersionReport> => {
      // Our own timer, held and cleared here, rather than AbortSignal.timeout, whose timer is
      // unref'd and belongs to a signal nothing here has a name for -- too much of somebody
      // else's bookkeeping for the one deadline every gated op waits behind. Cleared in the
      // `finally` below, so a context that gets its answer straight away leaves nothing pending.
      const deadline = new AbortController();
      const timer = setTimeout(
        () => deadline.abort(new Error(`the Galaxy version probe gave up after ${timeoutMs}ms`)),
        timeoutMs,
      );
      try {
        // The caller's cancellation and our own deadline both end the probe. Without them a
        // version route that never answers pins every gated op behind this one promise.
        const signal = AbortSignal.any(
          callerSignal ? [callerSignal, deadline.signal] : [deadline.signal],
        );
        const { data, error, response } = await Promise.race([
          client.GET("/api/version", { signal }),
          rejectWhenAborted(signal),
        ]);
        if (error || !data) {
          return { error: classifyHttp(response.status, error), source: "unknown" };
        }
        const payload = data as Record<string, unknown>;
        const major = payload["version_major"];
        const version = typeof major === "string" ? parseGalaxyVersion(major) : undefined;
        return { version, payload, source: version ? "server" : "unknown" };
      } catch (err) {
        // Unreachable, cancelled or too slow are all "unknown version", not a failed op -- but
        // a caller that has to report the server rather than guard against it needs the reason.
        return {
          error: new GalaxyConnectionError((err as Error).message, undefined, err),
          source: "unknown",
        };
      } finally {
        clearTimeout(timer);
      }
    })().then((report) => {
      // Dropped the moment it settles, so one aborted probe cannot leave later calls answering
      // "unknown" from a cached failure. The next call asks again.
      if (!report.payload) pending = undefined;
      return report;
    });
    return pending;
  };
}

export function createGalaxyContext(opts: CreateContextOptions): GalaxyContext {
  const client = createGalaxyClient(opts.baseUrl, opts.apiKey, opts.fetchImpl);
  return {
    client,
    baseUrl: opts.baseUrl,
    apiKey: opts.apiKey,
    serverVersion: opts.serverVersion,
    galaxyVersion: versionLookup(client, opts.serverVersion, opts.signal, opts.versionProbeTimeoutMs),
    poll: { ...DEFAULT_POLL, ...opts.poll },
    signal: opts.signal,
  };
}
