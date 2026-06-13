import { describe, it, expect } from "vitest";
import { readFileSync, existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import {
  DATASET_TERMINAL_STATES,
  JOB_MODEL_TERMINAL_STATES,
  INVOCATION_NON_TERMINAL_STATES,
} from "../src/terminal-states";

const MODEL = join(homedir(), "work/galaxy/lib/galaxy/model/__init__.py");
const gate = existsSync(MODEL) ? describe : describe.skip;

gate("terminal-state drift vs Galaxy source", () => {
  const src = existsSync(MODEL) ? readFileSync(MODEL, "utf8") : "";

  it("Dataset.terminal_states still matches", () => {
    // Extract the states.X members inside the terminal_states = ( ... ) tuple.
    const block = src.match(/terminal_states = \(([\s\S]*?)\)/)![1]!;
    const found = [...block.matchAll(/states\.([A-Z_]+)/g)].map((m) => m[1]!.toLowerCase());
    expect(new Set(found)).toEqual(new Set(DATASET_TERMINAL_STATES));
  });

  it("Job.terminal_states still matches", () => {
    // (?<!non_) so we don't accidentally match `non_terminal_states = [ ... ]`.
    const block = src.match(/(?<!non_)terminal_states = \[([\s\S]*?)\]/)![1]!;
    const found = [...block.matchAll(/states\.([A-Z_]+)/g)].map((m) => m[1]!.toLowerCase());
    expect(new Set(found)).toEqual(new Set(JOB_MODEL_TERMINAL_STATES));
  });

  it("WorkflowInvocation.non_terminal_states still matches", () => {
    const block = src.match(/non_terminal_states = \[([\s\S]*?)\]/)![1]!;
    const found = [...block.matchAll(/states\.([A-Z_]+)/g)].map((m) => m[1]!.toLowerCase());
    expect(new Set(found)).toEqual(new Set(INVOCATION_NON_TERMINAL_STATES));
  });
});
