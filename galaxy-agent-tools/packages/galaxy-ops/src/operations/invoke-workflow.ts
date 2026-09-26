import { z } from "zod";
import type { GalaxyContext } from "../context";
import { GalaxyConnectionError, GalaxyValidationError } from "../errors";
import { legacyGet } from "../legacy";
import { validateInputs, buildWorkflowInputTemplate, type DatatypesMapping } from "../workflow-inputs";
import { resolveWorkflowSlots } from "./get-workflow-input-template";
import { register, runOperation } from "./registry";
import type { AnyOperation, InputOf, Operation } from "./types";

// ---------------------------------------------------------------------------
// Datatype mapping fetch (port of Python _get_datatypes_mapping)
// No module-level memo: correctness doesn't require caching, and caching adds
// complexity + test surface. Callers that care about latency can add it later.
// ---------------------------------------------------------------------------

/**
 * Fetch the datatypes class mapping from GET /api/datatypes/types_and_mapping.
 * Returns the inner `datatypes_mapping` object: `{ ext_to_class_name, class_to_classes }`.
 * This path is in the typed bindings, so we use ctx.client.GET.
 */
export async function getDatatypesMapping(ctx: GalaxyContext): Promise<DatatypesMapping> {
  const { data, error } = await ctx.client.GET("/api/datatypes/types_and_mapping", {
    params: { query: { upload_only: false } },
  });
  const empty: DatatypesMapping = { ext_to_class_name: {}, class_to_classes: {} };
  if (error || !data) return empty;
  // The typed response is DatatypesCombinedMap; we want the inner datatypes_mapping.
  const combined = data as { datatypes_mapping?: DatatypesMapping };
  return combined.datatypes_mapping ?? empty;
}

// ---------------------------------------------------------------------------
// Enrich supplied inputs (port of Python _enrich_supplied_inputs)
// Best-effort: per-input fetch errors leave that entry un-enriched.
// ---------------------------------------------------------------------------

/**
 * For each supplied `{src, id}` input, fetch metadata the validator needs:
 * - hda -> attach `ext`
 * - hdca -> attach `collection_type` + sorted-unique `element_extensions`
 *
 * Errors on any single fetch are swallowed -- the validator is permissive
 * about missing metadata. Port of Python `_enrich_supplied_inputs`.
 */
async function enrichSuppliedInputs(
  ctx: GalaxyContext,
  inputs: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  const enriched: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(inputs)) {
    if (typeof value !== "object" || value === null || !("src" in (value as Record<string, unknown>))) {
      enriched[key] = value;
      continue;
    }
    const entry: Record<string, unknown> = { ...(value as Record<string, unknown>) };
    const ref = value as Record<string, unknown>;
    try {
      if (ref["src"] === "hda") {
        const id = ref["id"] as string;
        const { data } = await ctx.client.GET("/api/datasets/{dataset_id}", {
          params: { path: { dataset_id: id } },
        });
        if (data) {
          const ds = data as { extension?: string };
          entry["ext"] = ds.extension;
        }
      } else if (ref["src"] === "hdca") {
        const id = ref["id"] as string;
        const coll = await legacyGet<{
          collection_type?: string;
          elements?: Array<{ object?: { extension?: string } }>;
        }>(ctx, "/api/dataset_collections/{hdca_id}", {
          params: { path: { hdca_id: id } },
        });
        entry["collection_type"] = coll.collection_type;
        const exts = new Set<string>();
        for (const el of coll.elements ?? []) {
          const ext = el.object?.extension;
          if (ext) exts.add(ext);
        }
        entry["element_extensions"] = [...exts].sort();
      }
    } catch {
      // best-effort -- unknown metadata keeps the validator permissive
    }
    enriched[key] = entry;
  }
  return enriched;
}

// ---------------------------------------------------------------------------
// Invocation result type
// ---------------------------------------------------------------------------

export interface InvocationResult {
  id?: string;
  state?: string;
  [k: string]: unknown;
}

// ---------------------------------------------------------------------------
// Op
// ---------------------------------------------------------------------------

const DEFAULT_INPUTS_BY = "step_index";
const DEFAULT_PARAMETERS_NORMALIZED = false;

/** An object argument, or the JSON string some MCP clients send instead. */
const jsonObjectArg = z.union([z.record(z.string(), z.unknown()), z.string()]);

const input = {
  workflowId: z.string().describe("Encoded stored-workflow id (hexadecimal hash)"),
  inputs: jsonObjectArg
    .nullish()
    .describe(
      "Workflow inputs keyed by step_index. Each value is {src, id} for datasets/collections or a scalar for parameters. A JSON object string is accepted too.",
    ),
  params: jsonObjectArg
    .nullish()
    .describe(
      "Legacy step parameter overrides (use inputs for formal inputs instead). A JSON object string is accepted too.",
    ),
  historyId: z
    .string()
    .nullish()
    .describe("Encoded history id to store workflow outputs in"),
  historyName: z
    .string()
    .nullish()
    .describe("Name for a new history to create (ignored if historyId is provided)"),
  inputsBy: z
    .string()
    .default(DEFAULT_INPUTS_BY)
    .describe(
      "How inputs maps to workflow steps: 'step_index', 'step_uuid', 'name', or 'step_index|step_uuid'",
    ),
  parametersNormalized: z
    .boolean()
    .default(DEFAULT_PARAMETERS_NORMALIZED)
    .describe("Whether legacy parameters are already normalized (indexed by order_index)"),
};

type JsonObjectArg = Record<string, unknown> | string;

type In = {
  workflowId: string;
  inputs?: JsonObjectArg | null;
  params?: JsonObjectArg | null;
  historyId?: string | null;
  historyName?: string | null;
  inputsBy?: string;
  parametersNormalized?: boolean;
};

/**
 * Accept a nested argument as an object or as the JSON string clients serialize
 * it to. A blank string means "not supplied"; anything that is not a JSON object
 * fails here, naming the field, rather than obscurely deeper in.
 */
function coerceJsonObject(
  value: JsonObjectArg | null | undefined,
  name: string,
): Record<string, unknown> | undefined {
  if (value == null) return undefined;
  if (typeof value !== "string") {
    // The schema catches this on the MCP and CLI paths; a direct caller has nothing
    // in front of it, and a bare array reaching the POST body fails obscurely.
    if (typeof value !== "object" || Array.isArray(value)) {
      throw new GalaxyValidationError(
        `${name} must be a JSON object (a mapping) or a JSON object string, but got ${describeJson(value)}.`,
      );
    }
    return value;
  }
  if (!value.trim()) return undefined;
  let parsed: unknown;
  try {
    parsed = JSON.parse(value);
  } catch (err) {
    throw new GalaxyValidationError(
      `${name} must be a JSON object string or an object, but got invalid JSON: ${(err as Error).message}`,
    );
  }
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new GalaxyValidationError(
      `${name} must be a JSON object (a mapping), but the string parsed to ${describeJson(parsed)}.`,
    );
  }
  return parsed as Record<string, unknown>;
}

function describeJson(value: unknown): string {
  if (value === null) return "null";
  return Array.isArray(value) ? "an array" : `a ${typeof value}`;
}

async function run(i: In, ctx: GalaxyContext): Promise<InvocationResult> {
  const inputs = coerceJsonObject(i.inputs, "inputs");
  const params = coerceJsonObject(i.params, "params");

  // Preflight: only when inputs are provided and non-empty.
  if (inputs && Object.keys(inputs).length > 0) {
    // Wrap the whole preflight so that unexpected preflight failures (mapping
    // fetch error, slot resolution error, etc.) do NOT block a valid run.
    // Only a definitive reject from validateInputs blocks submission.
    // This matches the Python try/except scope in server.py:3082-3099.
    let rejects: unknown[] = [];
    let slots: Awaited<ReturnType<typeof resolveWorkflowSlots>>["slots"] = [];
    let warnings: unknown[] = [];
    try {
      const resolved = await resolveWorkflowSlots(ctx, i.workflowId, i.historyId ?? undefined);
      slots = resolved.slots;
      const mapping = await getDatatypesMapping(ctx);
      const enriched = await enrichSuppliedInputs(ctx, inputs);
      const report = validateInputs(slots, enriched, mapping);
      rejects = report.rejects;
      warnings = report.warnings;
    } catch {
      // preflight hiccup -- fall through to invoke
    }
    if (rejects.length > 0) {
      const template = buildWorkflowInputTemplate(slots, warnings, null, false);
      const lines = (rejects as Array<{ step_index: number; label?: string; reason: string }>)
        .map((r) => `  - step ${r.step_index} (${r.label ?? "?"}): ${r.reason}`)
        .join("\n");
      const hint =
        "\n\nExpected input slots (fill and retry with inputs_by='step_index|step_uuid'):\n" +
        JSON.stringify(template.slots, null, 2);
      throw new GalaxyValidationError(
        "Workflow inputs failed validation; not submitting:\n" + lines + hint,
      );
    }
  }

  // Build the POST body. Mirror only the fields Python/bioblend sends -- omitting
  // null-defaulted optional fields avoids accidentally overriding server defaults.
  // `history` carries "hist_id=<id>" when historyId is supplied, else the name, else omitted.
  const historyField = i.historyId
    ? `hist_id=${i.historyId}`
    : i.historyName
      ? i.historyName
      : undefined;

  const body: Record<string, unknown> = {
    inputs: inputs ?? {},
    inputs_by: i.inputsBy ?? DEFAULT_INPUTS_BY,
    parameters: params ?? {},
    parameters_normalized: i.parametersNormalized ?? DEFAULT_PARAMETERS_NORMALIZED,
  };
  if (historyField != null) body["history"] = historyField;

  const { data, error, response } = await ctx.client.POST(
    "/api/workflows/{workflow_id}/invocations",
    {
      params: { path: { workflow_id: i.workflowId } },
      body: body as never,
    },
  );

  if (error || !data) {
    const msg =
      error && typeof error === "object" && "err_msg" in error
        ? String((error as { err_msg: unknown }).err_msg)
        : `HTTP ${response.status}`;
    throw new GalaxyConnectionError(`Failed to invoke workflow: ${msg}`, response.status);
  }

  // The API can return a single invocation or an array (batch mode).
  // We return the first/only invocation dict.
  const result = Array.isArray(data) ? data[0] : data;
  return result as InvocationResult;
}

export const invokeWorkflowOp: Operation<typeof input, InvocationResult> = {
  name: "invoke_workflow",
  domain: "workflows",
  summary:
    "Invoke (run) a workflow with specified inputs and parameters. When inputs are provided, runs a preflight that validates them against the workflow's slots and the server's datatype hierarchy before submitting. Call get_workflow_input_template first to see the expected input shape.",
  input,
  readOnly: false,
  run,
  project: (out, i) => ({
    message: `Invoked workflow ${i.workflowId}${out.id ? ` (invocation ${out.id})` : ""}`,
  }),
};

register(invokeWorkflowOp as AnyOperation);

// The declared defaults reach run() only through whatever parsed the input -- MCP's
// registerTool, the CLI's safeParse -- and a programmatic caller supplies none of them, so
// they are filled in here rather than asserted away with a cast. The return type is the
// check: add a .default() above without one here and this stops compiling.
const withDefaults = (i: In): InputOf<typeof input> => ({
  ...i,
  inputsBy: i.inputsBy ?? DEFAULT_INPUTS_BY,
  parametersNormalized: i.parametersNormalized ?? DEFAULT_PARAMETERS_NORMALIZED,
});

export const invokeWorkflow = (i: In, ctx: GalaxyContext) =>
  runOperation(invokeWorkflowOp, withDefaults(i), ctx);
