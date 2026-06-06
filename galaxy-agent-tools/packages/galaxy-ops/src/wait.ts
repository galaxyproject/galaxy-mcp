import type { components } from "./bindings";
import type { GalaxyContext, PollPolicy } from "./context";
import { classifyHttp, JobFailedError } from "./errors";
import { isJobSuccess, isJobTerminal } from "./terminal-states";

export type JobDetail = components["schemas"]["EncodedJobDetails"];

function delay(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new Error("aborted"));
    const t = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(t);
      reject(new Error("aborted"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

function nextInterval(attempt: number, p: PollPolicy): number {
  const base = Math.min(p.intervalMs * p.backoff ** attempt, p.maxIntervalMs);
  const jitter = base * p.jitter * Math.random();
  return base + jitter;
}

/** Poll GET /api/jobs/{id} until the job reaches a terminal state. ok -> return; else throw. */
export async function waitForJob(jobId: string, ctx: GalaxyContext): Promise<JobDetail> {
  const start = Date.now();
  for (let attempt = 0; ; attempt++) {
    if (ctx.signal?.aborted) throw new Error("aborted");
    const { data, error, response } = await ctx.client.GET("/api/jobs/{job_id}", {
      params: { path: { job_id: jobId } },
    });
    if (error || !data) throw classifyHttp(response.status, error);
    const job = data as JobDetail;
    const state = String((job as { state: string }).state);
    if (isJobTerminal(state)) {
      if (isJobSuccess(state)) return job;
      // v2: fetch GET /api/jobs/{id}?full=true (ShowFullJobResponse) to populate stderr; the default detail has none.
      throw new JobFailedError(jobId, state);
    }
    if (Date.now() - start > ctx.poll.timeoutMs) {
      throw new Error(`Timed out waiting for job ${jobId} (last state=${state})`);
    }
    await delay(nextInterval(attempt, ctx.poll), ctx.signal);
  }
}
