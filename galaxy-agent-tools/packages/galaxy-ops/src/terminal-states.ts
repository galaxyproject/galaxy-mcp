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

// Job.terminal_states is exactly {ok, error, deleted}. "failed" is in the JobState enum
// but NOT in terminal_states; we still treat it as a terminal-FAILURE at runtime so the
// poll loop can't hang on it (see wait.ts).
export const JOB_MODEL_TERMINAL_STATES = ["ok", "error", "deleted"] as const;
export const JOB_TERMINAL_FAILURE_STATES = ["error", "deleted", "failed"] as const;
export const JOB_WAIT_TERMINAL_STATES = ["ok", "error", "deleted", "failed"] as const;

export const INVOCATION_NON_TERMINAL_STATES = ["new", "ready"] as const;
// "Truly finished": cancelled | failed | completed. "scheduled"/"cancelling" are in-flight.
export const INVOCATION_FINISHED_STATES = ["cancelled", "failed", "completed"] as const;

export function isJobTerminal(state: string): boolean {
  return (JOB_WAIT_TERMINAL_STATES as readonly string[]).includes(state);
}
export function isJobSuccess(state: string): boolean {
  return state === "ok";
}
