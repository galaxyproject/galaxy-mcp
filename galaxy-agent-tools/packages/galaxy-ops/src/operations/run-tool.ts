import { z } from "zod";
import type { GalaxyContext } from "../context";
import { jsonObject } from "../json-object";
import { legacyPost } from "../legacy";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

/**
 * What POST /api/tools answers a submission with: the jobs it queued and the datasets
 * they will write to, in the "new" state.
 *
 * Hand-typed: the classic tools endpoint is not in the OpenAPI bindings (see legacy.ts),
 * and this is Galaxy's own reply passed through rather than a shape built here.
 */
export interface ToolSubmission {
  outputs?: unknown[];
  output_collections?: unknown[];
  jobs?: unknown[];
  implicit_collections?: unknown[];
  [k: string]: unknown;
}

// An opaque record handed to Galaxy as-is in its legacy input format, which is what the
// other server sends: a parameter inside a section, conditional or repeat is one flat key
// with the path joined by "|" ("advanced|threshold"), not a nested object. Galaxy's legacy
// parser reads only the flat spelling, so a nested {advanced: {threshold}} is silently the
// default. The nested 21.01 shape belongs to the queue-and-wait path, `executeToolRequest`.
const input = {
  toolId: z.string().describe("Tool id, e.g. 'fastqc/0.74'"),
  historyId: z.string().describe("Encoded history id to run in"),
  inputs: jsonObject().describe(
    "Tool input parameters in Galaxy's legacy format: dataset inputs as " +
      "{\"input_name\": {\"src\": \"hda\", \"id\": \"dataset_id\"}}; a parameter inside a " +
      "section, conditional or repeat as one flat key joined with '|', e.g. " +
      "\"reference_source|ref_file\" -- not nested objects.",
  ),
  toolVersion: z.string().optional().describe("Optional explicit tool version"),
};

type RunToolInput = {
  toolId: string;
  historyId: string;
  inputs: Record<string, unknown>;
  toolVersion?: string;
};

/**
 * Submit the run and hand back what Galaxy said, which is what the other server does.
 *
 * POST /api/tools is the synchronous submit: it queues the jobs and answers with them
 * straight away, in the "new" state. It does not wait, and neither does this -- the
 * answer is a submission record, and a caller who wants the outputs polls the jobs. The
 * queue-and-wait path over /api/jobs and /api/tool_requests is still here as
 * `executeToolRequest` for a caller who wants it; it reports something else entirely
 * (`{toolRequestId, jobs, implicitCollections, state}`) and this op cannot both answer
 * with that and answer with what the other server answers with.
 *
 * `input_format: "legacy"` is what bioblend sends, and `tool_version` rides in the
 * payload rather than the query string -- Galaxy reads it out of the body in
 * services/tools.py `_create`.
 */
async function run(i: RunToolInput, ctx: GalaxyContext): Promise<ToolSubmission> {
  const body: Record<string, unknown> = {
    history_id: i.historyId,
    tool_id: i.toolId,
    input_format: "legacy",
    inputs: i.inputs,
  };
  if (i.toolVersion !== undefined) body["tool_version"] = i.toolVersion;
  return legacyPost<ToolSubmission>(ctx, "/api/tools", { body });
}

/** The jobs a submission queued, for the summary line. */
const jobCount = (o: ToolSubmission): number => (Array.isArray(o.jobs) ? o.jobs.length : 0);

export const runToolOp: Operation<typeof input, ToolSubmission> = {
  name: "run_tool", // parity: mcp-server-galaxy-py run_tool
  domain: "tools",
  summary:
    "Run a Galaxy tool. Inputs are Galaxy's legacy format: flat 'section|param' keys, not " +
    "nested objects. This queues the run and " +
    "answers with the jobs and output datasets Galaxy created, in their starting state -- " +
    "it does not wait for them; poll the jobs with get_job_details.",
  input,
  readOnly: false, // executes a tool -- not a read
  run,
  project: (o, i) => ({
    message: `Submitted ${i.toolId} to history ${i.historyId} (${jobCount(o)} job(s))`,
  }),
};

register(runToolOp as AnyOperation);

export const runTool = (i: RunToolInput, ctx: GalaxyContext) => runOperation(runToolOp, i, ctx);
