// Source of truth: ~/work/galaxy/lib/galaxy/model/__init__.py
//   Dataset.terminal_states            :4769
//   Job.terminal_states                :1770
//   WorkflowInvocation.non_terminal_states :9872
// Enums: lib/galaxy/schema/schema.py (JobState :128), schema/invocation.py (InvocationState :333)

export const DATASET_TERMINAL_STATES = [
  "ok",
  "empty",
  "error",
  "deferred",
  "discarded",
  "failed_metadata",
] as const;
// NOTE: "paused" and "new" are explicitly NOT terminal (no_data_states / non_ready_states).

// Job.terminal_states is exactly {ok, error, deleted}, which the drift test holds to the model.
export const JOB_MODEL_TERMINAL_STATES = ["ok", "error", "deleted"] as const;
/**
 * Job states that will not change again, read off `JobState` (lib/galaxy/schema/states.py):
 * the model's terminal_states (ok, error, deleted) plus failed, skipped -- a conditional step
 * that did not run -- and stopped. Every other state counts as still moving, so a state Galaxy
 * adds later keeps a wait open rather than calling it finished. The one list anything that waits
 * on a job settles on.
 */
export const JOB_SETTLED_STATES = ["ok", "skipped", "stopped", "error", "failed", "deleted"] as const;
/** The settled job states that are a failure. */
export const JOB_FAILED_STATES = ["error", "failed", "deleted"] as const;

export const INVOCATION_NON_TERMINAL_STATES = ["new", "ready"] as const;
// "Truly finished": cancelled | failed | completed. "scheduled"/"cancelling" are in-flight.
export const INVOCATION_FINISHED_STATES = ["cancelled", "failed", "completed"] as const;

export function isJobTerminal(state: string): boolean {
  return (JOB_SETTLED_STATES as readonly string[]).includes(state);
}
export function isJobSuccess(state: string): boolean {
  return state === "ok";
}
