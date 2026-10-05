/**
 * What an invocation amounts to once its jobs are counted.
 *
 * Galaxy's invocation `state` describes scheduling, not results: 26.x reports `completed` for a
 * run whose jobs failed, and `scheduled` while every job is still queued. The jobs summary
 * (`GET /api/invocations/{id}/jobs_summary`) says what happened, and this reads one off the other.
 */

import { JOB_FAILED_STATES, JOB_SETTLED_STATES } from "./terminal-states";

/** How many of an invocation's jobs are in each state, as jobs_summary counts them. */
export type JobStates = Record<string, number>;

/** Invocation states after which nothing more is scheduled. */
const SCHEDULING_DONE = new Set(["scheduled", "cancelled", "failed", "completed"]);

const counted = (jobStates: JobStates, wanted: (state: string) => boolean) =>
  Object.entries(jobStates).reduce((sum, [state, n]) => (n && wanted(state) ? sum + n : sum), 0);

/**
 * The outcome `state` amounts to: `failing` while a job has failed and others still move,
 * `failed` once everything settled with a failure, `cancelled` for a stop someone asked for,
 * `completed` when every job came back ok, and Galaxy's own state otherwise.
 *
 * A paused job does not move until someone resumes it. Galaxy pauses the jobs downstream of a
 * failed one, so a failure with paused jobs behind it is `failed`; paused jobs without a failure
 * leave Galaxy's own state, since the run is neither over nor going anywhere by itself.
 */
export function invocationOutcome(state: string | undefined, jobStates: JobStates): string | undefined {
  const failed = counted(jobStates, (s) => (JOB_FAILED_STATES as readonly string[]).includes(s));
  const paused = counted(jobStates, (s) => s === "paused");
  const active = counted(
    jobStates,
    (s) => s !== "paused" && !(JOB_SETTLED_STATES as readonly string[]).includes(s),
  );
  if (!SCHEDULING_DONE.has(state ?? "") || active) {
    return failed ? "failing" : state;
  }
  if (state === "cancelled") return "cancelled";
  if (failed || state === "failed") return "failed";
  if (paused) return state;
  return jobStates["ok"] ? "completed" : state;
}

/**
 * `invocation` with its jobs' states and the outcome they give it. Without a readable jobs
 * summary there is no outcome to give, and the invocation is answered as Galaxy sent it.
 */
export function withOutcome<T>(invocation: T, jobStates: JobStates | undefined): T {
  if (!jobStates || !invocation || typeof invocation !== "object" || Array.isArray(invocation)) {
    return invocation;
  }
  const record = invocation as Record<string, unknown>;
  const outcome = invocationOutcome(record["state"] as string | undefined, jobStates);
  return { ...record, job_states: jobStates, outcome } as T;
}
