import { z } from "zod";
import type { GalaxyContext } from "../context";
import { GalaxyError, GalaxyValidationError } from "../errors";
import { legacyPost } from "../legacy";
import { register, runOperation } from "./registry";
import type { AnyOperation, InputOf, Operation } from "./types";

/**
 * Hand-typed: what `POST /api/workflows` with `from_history_id` answers.
 *
 * server.py, extract_workflow_from_history: the same request through bioblend, the same
 * refusals word for word, and the same result. The 26.0.1 bindings type that path as `post?: never`, so this is read off the
 * controller instead (webapps/galaxy/api/workflows.py, `create`): the new
 * StoredWorkflow's `to_dict(view="collection")` -- id, latest_workflow_id, name,
 * create_time, update_time, published, importable, deleted, hidden,
 * latest_workflow_uuid, creator_deleted -- plus a `url`. No step count is in it.
 */
export interface ExtractedWorkflowRecord {
  id: string;
  name?: string | null;
  number_of_steps?: number;
  [k: string]: unknown;
}

export interface ExtractWorkflowResult {
  workflow_id: string;
  name: string | null;
  /** Galaxy's step count, when its reply carries one; it does not today. */
  number_of_steps: number | null;
  /** What was asked for, one step each: Galaxy makes a step per listed job and per listed input. */
  steps_requested: { tool_steps: number; dataset_inputs: number; collection_inputs: number };
  history_id: string;
  job_ids: string[];
  dataset_hids: number[];
  collection_hids: number[];
  /** Galaxy's record of the new stored workflow, as it answered. */
  workflow: ExtractedWorkflowRecord;
  /** Where to go next with the id, in one sentence. */
  next_steps: string;
}

/**
 * A history item number, as the caller may have it: an integer, or the digits a command
 * line hands over as a string.
 */
const hid = z.union([z.number().int(), z.string()]);

const input = {
  historyId: z.string().describe("Encoded id of the history to extract from"),
  workflowName: z.string().describe("Name for the new stored workflow"),
  jobIds: z
    .array(z.string())
    .optional()
    .describe(
      "Encoded ids of the jobs that become tool steps, in the order to extract them; get_job_details " +
        "and list_jobs report them. A job not from historyId is refused. Without any, the workflow has " +
        "no tool steps.",
    ),
  datasetHids: z
    .array(hid)
    .optional()
    .describe(
      "History item numbers (the hid get_history_contents lists, not encoded ids) of the datasets " +
        "that become the workflow's dataset inputs -- the ones the jobs read that another run would " +
        "replace. A dataset a listed job reads that is neither here nor produced by a listed job is " +
        "left as an unconnected step input.",
    ),
  collectionHids: z
    .array(hid)
    .optional()
    .describe(
      "History item numbers (hid) of the dataset collections that become the workflow's collection " +
        "inputs; a hid that is not a collection is refused by Galaxy.",
    ),
};
type In = {
  historyId: string;
  workflowName: string;
  jobIds?: string[];
  datasetHids?: (number | string)[];
  collectionHids?: (number | string)[];
};

/** Galaxy's payload for this way of creating a workflow; the names are its, not ours. */
export interface ExtractWorkflowPayload {
  from_history_id: string;
  workflow_name: string;
  job_ids: string[];
  dataset_ids: number[];
  dataset_collection_ids: number[];
}

function asHids(values: (number | string)[] | undefined, field: string): number[] {
  return (values ?? []).map((v) => {
    const n = typeof v === "string" ? (/^\d+$/.test(v.trim()) ? Number(v.trim()) : NaN) : v;
    if (!Number.isInteger(n) || n < 1) {
      throw new GalaxyValidationError(
        `${field} must be history item numbers (hid, a positive integer such as 3), got ${JSON.stringify(v)}. ` +
          "get_history_contents lists each item's hid. Nothing was sent to Galaxy.",
      );
    }
    return n;
  });
}

/**
 * What a caller has to send, checked before Galaxy sees it.
 *
 * Galaxy would accept an extraction that names nothing and make a workflow with no steps,
 * which nothing can be done with; refusing it here saves a stored workflow nobody wants.
 * The hids are checked because Galaxy's `dataset_ids` and `dataset_collection_ids` are hids
 * (workflow/extract.py reads them with `int()`), and an encoded id passed there would be
 * refused by Galaxy as a number it cannot parse rather than explained.
 *
 * The sentences are the Python server's, byte for byte, which is why they name the
 * parameters the way the MCP wire spells them; change both.
 */
export function extractWorkflowPayload(i: In): ExtractWorkflowPayload {
  const name = i.workflowName.trim();
  if (name === "") {
    throw new GalaxyValidationError("workflow_name must not be empty. Nothing was sent to Galaxy.");
  }
  const jobIds = i.jobIds ?? [];
  if (jobIds.some((id) => id.trim() === "")) {
    throw new GalaxyValidationError("job_ids must be encoded job ids; one of them is empty. Nothing was sent to Galaxy.");
  }
  const datasetHids = asHids(i.datasetHids, "dataset_hids");
  const collectionHids = asHids(i.collectionHids, "collection_hids");
  if (jobIds.length === 0 && datasetHids.length === 0 && collectionHids.length === 0) {
    throw new GalaxyValidationError(
      "Nothing to extract: name at least one job (job_ids) or one input (dataset_hids, collection_hids). " +
        "Galaxy would otherwise store a workflow with no steps. Nothing was sent to Galaxy.",
    );
  }
  return {
    from_history_id: i.historyId,
    workflow_name: name,
    job_ids: jobIds,
    dataset_ids: datasetHids,
    dataset_collection_ids: collectionHids,
  };
}

function nextSteps(workflowId: string): string {
  return (
    `To run it on other inputs, call get_workflow_input_template with workflow_id '${workflowId}' to see ` +
    "its input slots, fill them with the new datasets or collections, and invoke_workflow -- once " +
    "for every set of inputs, instead of re-running the tools one by one."
  );
}

async function run(i: In, ctx: GalaxyContext): Promise<ExtractWorkflowResult> {
  const body = extractWorkflowPayload(i);
  let record: ExtractedWorkflowRecord;
  try {
    record = await legacyPost<ExtractedWorkflowRecord>(ctx, "/api/workflows", { body });
  } catch (err) {
    // A 400 is Galaxy refusing what was named -- a hid that is not a collection, a job
    // that is not in this history, an id it cannot decode -- and the caller has to change
    // the call. As `connection` an agent would retry it and the CLI would exit 69 for a
    // usage error; see tool-input-error.ts for the same judgement on run_tool. The facts
    // stay on the error so the surface words it the way it words every other failure.
    if (err instanceof GalaxyError && err.http?.status === 400) {
      const refused = new GalaxyValidationError(err.message, { cause: err });
      refused.http = err.http;
      if (err.errCode !== undefined) refused.errCode = err.errCode;
      throw refused;
    }
    throw err;
  }
  const workflowId = String(record.id);
  return {
    workflow_id: workflowId,
    name: typeof record.name === "string" ? record.name : null,
    number_of_steps: typeof record.number_of_steps === "number" ? record.number_of_steps : null,
    steps_requested: {
      tool_steps: body.job_ids.length,
      dataset_inputs: body.dataset_ids.length,
      collection_inputs: body.dataset_collection_ids.length,
    },
    history_id: i.historyId,
    job_ids: body.job_ids,
    dataset_hids: body.dataset_ids,
    collection_hids: body.dataset_collection_ids,
    workflow: record,
    next_steps: nextSteps(workflowId),
  };
}

export const extractWorkflowFromHistoryOp: Operation<typeof input, ExtractWorkflowResult> = {
  name: "extract_workflow_from_history",
  domain: "workflows",
  summary:
    "Turn jobs already run in a history into a stored workflow, so the same chain of tools can " +
    "be invoked on other inputs instead of being re-run sample by sample. Name the history, a " +
    "name for the workflow, the jobs that become its tool steps (jobIds, encoded job ids) and " +
    "the datasets or collections that become its inputs (datasetHids and collectionHids, the " +
    "history item numbers). Each listed job and input becomes one step; a job's outputs feed the " +
    "later listed jobs that read them. Answers with the new workflow's id and name, what was " +
    "requested, and where to go next: get_workflow_input_template, then invoke_workflow.",
  input,
  readOnly: false,
  run,
  project: (d) => ({
    message:
      `Extracted workflow '${d.name ?? d.workflow_id}' (id ${d.workflow_id}) from history ${d.history_id}: ` +
      `${d.steps_requested.tool_steps} tool step(s) from the listed jobs, ${d.steps_requested.dataset_inputs} ` +
      `dataset input(s) and ${d.steps_requested.collection_inputs} collection input(s). ${d.next_steps}`,
  }),
  // The one request is a POST, worded as a bioblend write with the history and the name in
  // the context; Galaxy's own refusals (a hid that is not a collection, a job not in the
  // history) arrive in its err_msg and are quoted by that sentence.
  failure: {
    shape: "bioblend-write",
    action: "Extract workflow from history",
    context: (i) => ({ history_id: i.historyId, workflow_name: i.workflowName }),
  },
};

register(extractWorkflowFromHistoryOp as AnyOperation);

export const extractWorkflowFromHistory = (i: In, ctx: GalaxyContext) =>
  runOperation(extractWorkflowFromHistoryOp, i as InputOf<typeof input>, ctx);
