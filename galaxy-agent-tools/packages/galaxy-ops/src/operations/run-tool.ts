import { z } from "zod";
import type { GalaxyContext } from "../context";
import { jsonObject } from "../json-object";
import { legacyPost } from "../legacy";
import { enrichedRunFailure } from "../tool-input-error";
import { preflightToolInputs, toolCredentialsContext } from "../tool-preflight";
import { envelopeFact, readFact, recordFact } from "./envelope-facts";
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
  toolVersion: z.string().nullish().describe("Optional explicit tool version"),
};

type RunToolInput = {
  toolId: string;
  historyId: string;
  inputs: Record<string, unknown>;
  toolVersion?: string | null;
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
/** Why the inputs were not checked against the tool's schema, when they were not. */
const uncheckedInputs = envelopeFact<string>("run_tool.unchecked_inputs");
/** Whether stored credentials were found and sent with the run. */
const withCredentials = envelopeFact<boolean>("run_tool.with_credentials");

async function run(i: RunToolInput, ctx: GalaxyContext): Promise<ToolSubmission> {
  // Before anything is sent, and in this order, because that is the order over there: the
  // schema read only when an input looks like a dataset, then the credentials lookup, then
  // the run. The first two both decide a clause of the sentence the run answers with.
  const unchecked = await preflightToolInputs(ctx, i.toolId, i.inputs, {
    toolVersion: i.toolVersion,
  });
  if (unchecked !== null) recordFact(ctx, uncheckedInputs, unchecked);

  const credentials = await toolCredentialsContext(ctx, i.toolId);
  if (credentials !== null) recordFact(ctx, withCredentials, true);

  const body: Record<string, unknown> = {
    history_id: i.historyId,
    tool_id: i.toolId,
    input_format: "legacy",
    inputs: i.inputs,
  };
  if (i.toolVersion != null) body["tool_version"] = i.toolVersion;
  if (credentials !== null) body["credentials_context"] = credentials;
  try {
    return await legacyPost<ToolSubmission>(ctx, "/api/tools", { body });
  } catch (err) {
    // A refused run gets the other server's two enriched sentences where they apply: a
    // failure that mentions credentials, and a 400, which is the status Galaxy refuses a
    // tool form with. Anything else is left to the op's failure contract below.
    throw await enrichedRunFailure(ctx, err, {
      action: "Run tool",
      toolId: i.toolId,
      historyId: i.historyId,
      inputs: i.inputs,
      credentials: { used: credentials !== null },
      toolVersion: i.toolVersion,
    });
  }
}

/**
 * The version that ran, when every job Galaxy returned names the same one.
 *
 * Port of `_reported_tool_version` in server.py, and its reasoning is the load-bearing
 * part: POST /api/tools serialises each job with a visible `tool_version`, so the reply
 * says what ran, while ASKING for a version does not -- Galaxy's toolbox hands back the
 * newest installed version when the requested one is missing. One job that names no
 * version, two that disagree, or a reply with no jobs all leave the provenance unknown,
 * and there is no partly-known version to report, so the answer is null and the sentence
 * says so.
 */
function reportedToolVersion(result: ToolSubmission): string | null {
  const jobs = result.jobs;
  if (!Array.isArray(jobs) || jobs.length === 0) return null;
  const versions = new Set<string>();
  for (const job of jobs) {
    const version =
      job !== null && typeof job === "object"
        ? (job as { tool_version?: unknown }).tool_version
        : undefined;
    if (typeof version !== "string" || version === "") return null;
    versions.add(version);
  }
  return versions.size === 1 ? [...versions][0]! : null;
}

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
  project: (o, i, facts) => {
    // server.py, run_tool: the tool and the history, with a clause about the version
    // only when one was asked for -- and that clause reports what the jobs say ran,
    // never the request repeated back as if it were the answer. Then the credentials
    // clause and the unchecked-inputs clause, in that order, each from a fact the run
    // left behind.
    let version = "";
    if (i.toolVersion != null) {
      const ran = reportedToolVersion(o);
      if (ran === null) version = ` at an unreported version (${i.toolVersion} requested)`;
      else if (ran === i.toolVersion) version = ` at version ${ran}`;
      else version = ` at version ${ran} (not the ${i.toolVersion} requested)`;
    }
    const credentials = readFact(facts, withCredentials) ? " (with credentials)" : "";
    const unchecked = readFact(facts, uncheckedInputs);
    const notChecked = unchecked ? ` (inputs not pre-checked: ${unchecked})` : "";
    return {
      message:
        `Started tool '${i.toolId}'${version} in history ` +
        `'${i.historyId}'${credentials}${notChecked}`,
    };
  },
  // server.py, run_tool: the run is a bioblend write. A 400 is the status Galaxy rejects a
  // tool form with, and that failure gets the input-shape explanation rather than the bare
  // sentence -- see the throw site.
  failure: {
    shape: "bioblend-write",
    action: "Run tool",
    context: (i) => ({ history_id: i.historyId, tool_id: i.toolId, inputs: i.inputs }),
  },
};

register(runToolOp as AnyOperation);

export const runTool = (i: RunToolInput, ctx: GalaxyContext) => runOperation(runToolOp, i, ctx);
