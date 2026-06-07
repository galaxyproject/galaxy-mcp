import { createGalaxyClient, type GalaxyClient } from "./client";

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

export interface GalaxyContext {
  readonly client: GalaxyClient;
  readonly baseUrl?: string;
  readonly serverVersion?: string;
  readonly poll: PollPolicy;
  readonly signal?: AbortSignal;
}

export interface CreateContextOptions {
  baseUrl: string;
  apiKey: string;
  serverVersion?: string;
  poll?: Partial<PollPolicy>;
  signal?: AbortSignal;
  fetchImpl?: typeof fetch; // tests only
}

export function createGalaxyContext(opts: CreateContextOptions): GalaxyContext {
  return {
    client: createGalaxyClient(opts.baseUrl, opts.apiKey, opts.fetchImpl),
    baseUrl: opts.baseUrl,
    serverVersion: opts.serverVersion,
    poll: { ...DEFAULT_POLL, ...opts.poll },
    signal: opts.signal,
  };
}
