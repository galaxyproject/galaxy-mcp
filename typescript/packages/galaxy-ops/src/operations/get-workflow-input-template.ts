import { z } from "zod";
import type { GalaxyContext } from "../context";
import { GalaxyValidationError, httpError } from "../errors";
import { legacyGet } from "../legacy";
import { envelopeFact, readFact, recordFact } from "./envelope-facts";
import {
  normalizeRunModel,
  normalizeGaSteps,
  findLegacyWarnings,
  buildGuide,
  buildWorkflowInputTemplate,
  type WorkflowInputTemplate,
  type WorkflowSlot,
} from "../workflow-inputs";
import { register, runOperation } from "./registry";
import type { AnyOperation, InputOf, Operation } from "./types";

// ---------------------------------------------------------------------------
// Types for the off-schema endpoints
// ---------------------------------------------------------------------------

/** style=run download -- not in the typed bindings */
interface RunModelDict {
  steps?: unknown;
  has_upgrade_messages?: unknown;
  step_version_changes?: unknown;
  [k: string]: unknown;
}

/** .ga export or show_workflow -- also not fully typed */
interface WorkflowDict {
  steps?: unknown;
  [k: string]: unknown;
}

// ---------------------------------------------------------------------------
// resolveWorkflowSlots -- shared with Wave F3 (invoke_workflow)
// ---------------------------------------------------------------------------

/**
 * Which of the two sources the slots came from, for the summary line.
 *
 * `run()` returns the template and not how it was built, and the other server's
 * sentence names the source ("source: style=run"), so it travels beside the call the
 * way list_pages' total does.
 */
const slotProvenance = envelopeFact<"style=run" | "ga-fallback">(
  "get_workflow_input_template.provenance",
);

export interface ResolvedSlots {
  slots: WorkflowSlot[];
  provenance: "style=run" | "ga-fallback";
  /** The parsed style=run dict when that path was used, else null. */
  runModel: RunModelDict | null;
}

/**
 * Resolve a workflow's input slots. Primary: style=run (webapp's source),
 * behind normalizeRunModel. Fallback: the .ga export via normalizeGaSteps.
 *
 * Port of Python `_resolve_workflow_slots` (server.py:2930-2955).
 * Exported so Wave F3 (invoke_workflow) can reuse it without re-fetching.
 *
 * `instance=false` is load-bearing: workflow_id is a StoredWorkflow id; instance=true
 * would reinterpret it as a Workflow-version id and template the wrong inputs.
 */
export async function resolveWorkflowSlots(
  ctx: GalaxyContext,
  workflowId: string,
  historyId?: string | null,
  version?: number | null,
): Promise<ResolvedSlots> {
  // Primary: style=run -- off-schema endpoint, use legacyGet
  try {
    const query: Record<string, unknown> = { style: "run", instance: false };
    if (historyId) query["history_id"] = historyId;
    if (version != null) query["version"] = version;

    const runModel = await legacyGet<RunModelDict>(
      ctx,
      "/api/workflows/{workflow_id}/download",
      { params: { path: { workflow_id: workflowId }, query } },
    );
    const slots = normalizeRunModel(runModel as Record<string, unknown>);
    if (slots.length > 0) {
      return { slots, provenance: "style=run", runModel };
    }
  } catch {
    // style=run unavailable or returned an error -- fall through to .ga fallback
  }

  // Fallback: .ga export (no style param), pinned to the same version so the slots
  // never come from a different version than the run model was asked for.
  const definition = await legacyGet<WorkflowDict>(
    ctx,
    "/api/workflows/{workflow_id}/download",
    { params: { path: { workflow_id: workflowId }, query: versionQuery(version) } },
  );
  const slots = normalizeGaSteps(definition as Record<string, unknown>);
  return { slots, provenance: "ga-fallback", runModel: null };
}

/**
 * `{ version }` when one was asked for, nothing at all otherwise -- bioblend adds the key
 * only when it is not None, and the request has to read the same way on both sides.
 */
function versionQuery(version: number | null | undefined): { version: number } | undefined {
  return version == null ? undefined : { version };
}

// ---------------------------------------------------------------------------
// Op
// ---------------------------------------------------------------------------

const input = {
  workflowId: z.string().describe("Encoded stored-workflow id"),
  historyId: z
    .string()
    .nullish()
    .describe(
      "History id; resolves history-compatible dataset options in the run model",
    ),
  verbose: z
    .boolean()
    .default(false)
    .describe("Return the full readme and uncapped option lists (default false)"),
  version: z
    .number()
    .int()
    .nullish()
    .describe(
      "Stored version to template, counted as get_workflow_details counts them (0 is the oldest); the latest when left out",
    ),
};
type In = {
  workflowId: string;
  historyId?: string | null;
  verbose?: boolean;
  version?: number | null;
};

async function run(i: In, ctx: GalaxyContext): Promise<WorkflowInputTemplate> {
  const verbose = i.verbose ?? false;
  const version = i.version ?? null;

  // server.py, get_workflow_input_template: refused before anything is sent. Galaxy
  // picks a stored version with a plain list subscript, so a negative index is not a 400
  // there but some other version, served quietly.
  if (version !== null && version < 0) {
    throw new GalaxyValidationError(
      `version must be 0 or greater (got ${version}); 0 is the oldest stored ` +
        "version and get_workflow_details counts up from there",
    );
  }

  // Three independent best-effort reads of the same workflow (mirrors Python), every
  // one pinned to `version` when there is one.
  const { slots, provenance, runModel } = await resolveWorkflowSlots(
    ctx,
    i.workflowId,
    i.historyId,
    version,
  );
  recordFact(ctx, slotProvenance, provenance);

  // .ga export for legacy warnings (best-effort)
  let warnings: Array<{ kind: string; message: string }> = [];
  try {
    const definition = await legacyGet<WorkflowDict>(
      ctx,
      "/api/workflows/{workflow_id}/download",
      { params: { path: { workflow_id: i.workflowId }, query: versionQuery(version) } },
    );
    warnings = findLegacyWarnings(definition as Record<string, unknown>);
  } catch {
    // best-effort -- warnings absent is fine
  }

  // show_workflow for guide docs (best-effort)
  let workflowShow: Record<string, unknown> = {};
  try {
    const { data, error, response } = await ctx.client.GET("/api/workflows/{workflow_id}", {
      params: { path: { workflow_id: i.workflowId }, query: versionQuery(version) },
    });
    if (!error && data) workflowShow = data as Record<string, unknown>;
    else if (error) throw httpError(response, error);
  } catch {
    // best-effort -- guide absent is fine
  }

  const guide = buildGuide(workflowShow, runModel as Record<string, unknown> | null, verbose);
  return buildWorkflowInputTemplate(slots, warnings, guide, verbose);
}

export const getWorkflowInputTemplateOp: Operation<typeof input, WorkflowInputTemplate> = {
  name: "get_workflow_input_template",
  domain: "workflows",
  summary:
    "Return a ready-to-fill input template plus a run guide for a workflow. Call this before invoke_workflow. Each slot lists its label, expected src (hda/hdca), accepted datatypes, collection type, and -- for parameters -- selectable options.",
  input,
  run,
  project: (out, i, facts) => {
    // A caller that projected by hand left no fact behind, and the answer is still in
    // the output: buildGuide adds `notes` exactly when there was no run model, which
    // is exactly the .ga fallback path.
    const guide = out.guide as { notes?: unknown } | undefined;
    const provenance =
      readFact(facts, slotProvenance) ??
      (guide && "notes" in guide ? "ga-fallback" : "style=run");
    const slots = (out.slots as unknown[]).length;
    return {
      // server.py, get_workflow_input_template, sentence for sentence -- including the
      // "(s)" this one really does write, and the quoted inputs_by hint.
      message:
        `Built an input template for workflow '${i.workflowId}' ` +
        `(${slots} slot(s), source: ${provenance}). Fill inputs_template ` +
        "and invoke with inputs_by='step_index|step_uuid'.",
      count: slots,
    };
  },
  // server.py, get_workflow_input_template: only the export fallback can fail there -- the
  // run-model fetch and the two best-effort reads are swallowed on both sides.
  // The version joins the context only when one was asked for, so an unpinned call fails
  // in exactly the words it did before the parameter existed.
  failure: {
    shape: "bioblend-get",
    action: "Get workflow input template",
    context: (i) =>
      i.version == null
        ? { workflow_id: i.workflowId }
        : { workflow_id: i.workflowId, version: i.version },
  },
};

register(getWorkflowInputTemplateOp as AnyOperation);

// A library caller may leave the defaulted arguments out; run() applies the same
// values the schema declares for the parsed surface path.
export const getWorkflowInputTemplate = (i: In, ctx: GalaxyContext) =>
  runOperation(getWorkflowInputTemplateOp, i as InputOf<typeof input>, ctx);
