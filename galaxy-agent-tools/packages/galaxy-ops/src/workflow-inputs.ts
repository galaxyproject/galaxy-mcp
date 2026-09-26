/**
 * Pure helpers for workflow input templates and invoke-time validation.
 *
 * No HTTP, no context, no side effects -- all functions take plain parsed JSON
 * so they are trivially unit-testable. The I/O wiring (fetching style=run / .ga /
 * show_workflow via a Galaxy client, op registration) lives in
 * operations/get-workflow-input-template.ts and operations/invoke-workflow.ts.
 *
 * Faithful port of mcp-server-galaxy-py/src/galaxy_mcp/ops/workflow_inputs.py.
 */

import { cleanReadmeSummary } from "./iwc-manifest";

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/** Best-effort int coercion; returns undefined instead of throwing. */
function safeInt(value: unknown): number | undefined {
  if (value == null) return undefined;
  const n = Number(value);
  return Number.isInteger(n) ? n : undefined;
}

/** Coerce a possibly-scalar field to a list. A bare string becomes a single-element list. */
function asList(value: unknown): unknown[] {
  if (value == null) return [];
  if (Array.isArray(value)) return value;
  return [value];
}

/** Parse a JSON-encoded tool_state string, or pass through dicts. */
function coerceState(toolState: unknown): Record<string, unknown> {
  if (typeof toolState === "string") {
    try {
      const parsed = JSON.parse(toolState);
      return typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)
        ? (parsed as Record<string, unknown>)
        : {};
    } catch {
      return {};
    }
  }
  if (typeof toolState === "object" && toolState !== null && !Array.isArray(toolState)) {
    return toolState as Record<string, unknown>;
  }
  return {};
}

// Sentinel value for sorting non-numeric step keys to the end.
const SORT_SENTINEL = 1e9;

const INPUT_TYPE_MAP: Record<string, string> = {
  data_input: "data",
  data_collection_input: "data_collection",
  parameter_input: "parameter",
};

const SRC_MAP: Record<string, string | null> = {
  data: "hda",
  data_collection: "hdca",
  parameter: null,
};

const FALLBACK_LABEL: Record<string, string> = {
  data: "Input dataset",
  data_collection: "Input dataset collection",
  parameter: "Input parameter",
};

// ---------------------------------------------------------------------------
// Slot types
// ---------------------------------------------------------------------------

export interface WorkflowSlot {
  step_index: number;
  step_uuid: string | null | undefined;
  label: string;
  input_type: string;
  src: string | null;
  accepted_formats: unknown[];
  acceptable_extensions: unknown[];
  collection_type: unknown;
  parameter_type: unknown;
  optional: boolean;
  options: Array<{ label: string; value: string }>;
}

// ---------------------------------------------------------------------------
// Slot builder (shared by both normalizers)
// ---------------------------------------------------------------------------

function makeSlot(args: {
  step_index: number;
  step_uuid: string | null | undefined;
  label: string;
  input_type: string;
  accepted_formats: unknown[];
  acceptable_extensions: unknown[];
  collection_type: unknown;
  parameter_type: unknown;
  optional: boolean;
  options: Array<{ label: string; value: string }>;
}): WorkflowSlot {
  return {
    step_index: args.step_index,
    step_uuid: args.step_uuid,
    label: args.label,
    input_type: args.input_type,
    src: SRC_MAP[args.input_type] ?? null,
    accepted_formats: args.accepted_formats,
    acceptable_extensions: args.acceptable_extensions,
    collection_type: args.collection_type,
    parameter_type: args.parameter_type,
    optional: args.optional,
    options: args.options,
  };
}

// ---------------------------------------------------------------------------
// Options helpers
// ---------------------------------------------------------------------------

/**
 * style=run param options come as [label, value, selected] triples.
 */
function optionsFromTriples(raw: unknown): Array<{ label: string; value: string }> {
  if (!Array.isArray(raw)) return [];
  const out: Array<{ label: string; value: string }> = [];
  for (const item of raw) {
    if (Array.isArray(item) && item.length >= 2) {
      out.push({ label: String(item[0]), value: String(item[1]) });
    }
  }
  return out;
}

/**
 * .ga enumerated params carry a flat list of allowed string values.
 */
function optionsFromRestrictions(raw: unknown): Array<{ label: string; value: string }> {
  if (!Array.isArray(raw)) return [];
  return raw.map((v) => ({ label: String(v), value: String(v) }));
}

// ---------------------------------------------------------------------------
// normalize_ga_steps
// ---------------------------------------------------------------------------

/**
 * Normalize a .ga / IWC-manifest workflow definition into input slots.
 *
 * Reads `definition.steps` (dict keyed by string order_index), keeps only
 * data_input / data_collection_input / parameter_input steps, and parses each
 * step's `tool_state` for declared constraints. Absent `format` means
 * "no restriction" (empty accepted_formats).
 *
 * Faithful port of Python `normalize_ga_steps`.
 */
export function normalizeGaSteps(definition: Record<string, unknown>): WorkflowSlot[] {
  const steps = definition["steps"];
  if (typeof steps !== "object" || steps == null || Array.isArray(steps)) return [];

  const stepsRecord = steps as Record<string, unknown>;

  // Sort numeric keys first (ascending), non-numeric keys to the end (then skipped).
  const sorted = Object.entries(stepsRecord).sort(([ka], [kb]) => {
    const ia = safeInt(ka) ?? SORT_SENTINEL;
    const ib = safeInt(kb) ?? SORT_SENTINEL;
    if (ia !== ib) return ia - ib;
    return ka < kb ? -1 : ka > kb ? 1 : 0;
  });

  const slots: WorkflowSlot[] = [];
  for (const [key, step] of sorted) {
    const index = safeInt(key);
    if (index == null) continue; // non-numeric key -- skip

    const stepObj = typeof step === "object" && step !== null ? (step as Record<string, unknown>) : {};
    const inputType = INPUT_TYPE_MAP[String(stepObj["type"] ?? "")];
    if (!inputType) continue;

    const state = coerceState(stepObj["tool_state"]);
    const labelFallback = `${FALLBACK_LABEL[inputType]} (step ${index})`;
    const label = String(stepObj["label"] ?? "") || labelFallback;

    slots.push(
      makeSlot({
        step_index: index,
        step_uuid: stepObj["uuid"] as string | undefined,
        label,
        input_type: inputType,
        accepted_formats: asList(state["format"]),
        acceptable_extensions: [],
        collection_type: state["collection_type"] ?? null,
        parameter_type: state["parameter_type"] ?? null,
        optional: Boolean(state["optional"] ?? false),
        options:
          inputType === "parameter" ? optionsFromRestrictions(state["restrictions"]) : [],
      }),
    );
  }
  return slots;
}

// ---------------------------------------------------------------------------
// normalize_run_model
// ---------------------------------------------------------------------------

/**
 * Normalize a style=run workflow model into input slots (the slot contract).
 *
 * style=run is the webapp's own run-form serialization; for data inputs its
 * `extensions` already reflect Galaxy's downstream-consumer resolution.
 *
 * Faithful port of Python `normalize_run_model`.
 */
export function normalizeRunModel(runDict: Record<string, unknown>): WorkflowSlot[] {
  const rawSteps = runDict["steps"];

  let stepIter: unknown[];
  if (typeof rawSteps === "object" && rawSteps !== null && !Array.isArray(rawSteps)) {
    // dict keyed by step index -- sort numerically, non-numeric keys to end
    stepIter = Object.entries(rawSteps as Record<string, unknown>)
      .sort(([ka], [kb]) => {
        const ia = safeInt(ka) ?? SORT_SENTINEL;
        const ib = safeInt(kb) ?? SORT_SENTINEL;
        if (ia !== ib) return ia - ib;
        return ka < kb ? -1 : ka > kb ? 1 : 0;
      })
      .map(([, v]) => v);
  } else {
    stepIter = Array.isArray(rawSteps) ? rawSteps : [];
  }

  const slots: WorkflowSlot[] = [];
  for (const step of stepIter) {
    const stepObj =
      typeof step === "object" && step !== null ? (step as Record<string, unknown>) : {};

    // style=run uses step_type; .ga fallback path uses type
    const inputType =
      INPUT_TYPE_MAP[String(stepObj["step_type"] ?? "")] ??
      INPUT_TYPE_MAP[String(stepObj["type"] ?? "")];
    if (!inputType) continue;

    // style=run can expose step_index, order_index, or id; skip if none are numeric
    const rawIdx =
      stepObj["step_index"] ?? stepObj["order_index"] ?? stepObj["id"];
    const index = safeInt(rawIdx);
    if (index == null) continue;

    // The run model nests the actual param under "inputs"[0] for input steps.
    const inputs = Array.isArray(stepObj["inputs"]) ? stepObj["inputs"] : [];
    const param =
      typeof inputs[0] === "object" && inputs[0] !== null
        ? (inputs[0] as Record<string, unknown>)
        : {};

    // collection_type: prefer param.collection_type, fallback to first of param.collection_types
    let ctype: unknown = param["collection_type"] ?? null;
    if (!ctype) {
      const ctypes = param["collection_types"];
      ctype = Array.isArray(ctypes) && ctypes.length > 0 ? ctypes[0] : null;
    }

    // style=run uses "step_label" for the step name; param.label is a fallback
    const labelFallback = `${FALLBACK_LABEL[inputType]} (step ${index})`;
    const label =
      String(stepObj["step_label"] ?? "") ||
      String(param["label"] ?? "") ||
      labelFallback;

    // parameter_type: style=run puts it on param first, then step
    const parameterType = param["parameter_type"] ?? stepObj["parameter_type"] ?? null;

    slots.push(
      makeSlot({
        step_index: index,
        step_uuid: stepObj["uuid"] as string | undefined,
        label,
        input_type: inputType,
        accepted_formats: asList(param["extensions"]),
        acceptable_extensions: asList(param["acceptable_extensions"]),
        collection_type: ctype,
        parameter_type: parameterType,
        optional: Boolean(param["optional"] ?? false),
        options: inputType === "parameter" ? optionsFromTriples(param["options"]) : [],
      }),
    );
  }
  return slots;
}

// ---------------------------------------------------------------------------
// find_legacy_warnings
// ---------------------------------------------------------------------------

function hasRuntimeValue(obj: unknown): boolean {
  if (typeof obj !== "object" || obj === null) return false;
  if (Array.isArray(obj)) return obj.some(hasRuntimeValue);
  const rec = obj as Record<string, unknown>;
  if (rec["__class__"] === "RuntimeValue") return true;
  return Object.values(rec).some(hasRuntimeValue);
}

/**
 * Warn on tool steps carrying unconnected RuntimeValue params (the real
 * legacy-run-form trigger). Bare `parameter_input` steps are formal inputs
 * and are NOT flagged.
 *
 * Faithful port of Python `find_legacy_warnings`.
 */
export function findLegacyWarnings(
  definition: Record<string, unknown>,
): Array<{ kind: string; message: string }> {
  const steps = definition["steps"];
  if (typeof steps !== "object" || steps == null || Array.isArray(steps)) return [];

  const stepsRecord = steps as Record<string, unknown>;
  const sorted = Object.entries(stepsRecord).sort(([ka], [kb]) => {
    const ia = safeInt(ka) ?? SORT_SENTINEL;
    const ib = safeInt(kb) ?? SORT_SENTINEL;
    if (ia !== ib) return ia - ib;
    return ka < kb ? -1 : ka > kb ? 1 : 0;
  });

  const warnings: Array<{ kind: string; message: string }> = [];
  for (const [key, step] of sorted) {
    const stepObj =
      typeof step === "object" && step !== null ? (step as Record<string, unknown>) : {};
    if (stepObj["type"] !== "tool") continue;
    if (hasRuntimeValue(coerceState(stepObj["tool_state"]))) {
      const name = String(stepObj["label"] ?? stepObj["tool_id"] ?? `step ${key}`);
      warnings.push({
        kind: "legacy_runtime_value",
        message: `Tool step '${name}' has a RuntimeValue parameter set at runtime (legacy run-form pattern); the workflow may not run cleanly via the API.`,
      });
    }
  }
  return warnings;
}

// ---------------------------------------------------------------------------
// build_workflow_input_template
// ---------------------------------------------------------------------------

const OPTIONS_INLINE_CAP = 25;
const OPTIONS_SAMPLE = 15;

function placeholderFor(slot: WorkflowSlot): unknown {
  if (slot.input_type === "data") return { src: "hda", id: "<dataset_id>" };
  if (slot.input_type === "data_collection") return { src: "hdca", id: "<collection_id>" };
  return "<value>";
}

function displaySlot(slot: WorkflowSlot, verbose: boolean): Record<string, unknown> {
  // Drop acceptable_extensions (implementation detail, can be large)
  const { acceptable_extensions: _ae, ...rest } = slot;
  const out: Record<string, unknown> = { ...rest };

  const options = (out["options"] as unknown[]) ?? [];
  if (!options.length) {
    delete out["options"];
    return out;
  }
  out["option_count"] = options.length;
  if (!verbose && options.length > OPTIONS_INLINE_CAP) {
    out["options"] = options.slice(0, OPTIONS_SAMPLE);
    out["options_note"] = `showing ${OPTIONS_SAMPLE} of ${options.length}; pass verbose=true for the full list, or supply a value matching an installed option.`;
  }
  return out;
}

export interface WorkflowInputTemplate {
  inputs_template: Record<string, unknown>;
  slots: Record<string, unknown>[];
  inputs_by: string;
  warnings: unknown[];
  guide?: Record<string, unknown>;
}

/**
 * Assemble the model-facing template: a ready-to-fill skeleton keyed by
 * step_index, the per-slot constraint summary (with capped options), the
 * invoke key hint, any legacy warnings, and -- when provided -- the run guide.
 *
 * Faithful port of Python `build_workflow_input_template`.
 */
export function buildWorkflowInputTemplate(
  slots: WorkflowSlot[],
  warnings: unknown[] | null | undefined,
  guide: Record<string, unknown> | null | undefined,
  verbose: boolean,
): WorkflowInputTemplate {
  const result: WorkflowInputTemplate = {
    inputs_template: Object.fromEntries(slots.map((s) => [String(s.step_index), placeholderFor(s)])),
    slots: slots.map((s) => displaySlot(s, verbose)),
    inputs_by: "step_index|step_uuid",
    warnings: warnings ?? [],
  };
  if (guide != null) {
    result.guide = guide;
  }
  return result;
}

// ---------------------------------------------------------------------------
// build_guide
// ---------------------------------------------------------------------------

export interface WorkflowGuide {
  summary: string;
  annotation: string;
  provenance: {
    version: unknown;
    source: { trs_id: unknown; trs_url: unknown };
    freshness?: {
      has_upgrade_messages: unknown;
      step_version_changes: unknown[];
    };
  };
  notes?: string[];
  [k: string]: unknown;
}

/**
 * Assemble the model-facing run guide from a show_workflow dict.
 *
 * summary: full readme when verbose, else a cleaned summary; falls back
 * readme -> help -> annotation. provenance: version + TRS source (always),
 * plus freshness flags from style=run when available. When runModel is null
 * (the .ga fallback path), parameter options weren't resolved -- note it.
 *
 * Faithful port of Python `build_guide`.
 */
export function buildGuide(
  workflowShow: Record<string, unknown>,
  runModel: Record<string, unknown> | null | undefined,
  verbose: boolean,
): WorkflowGuide {
  const readme = String(workflowShow["readme"] ?? "");
  const helpText = String(workflowShow["help"] ?? "");
  const annotation = String(workflowShow["annotation"] ?? "");

  // Pick the first source whose cleaned form has real prose (headers-only readme
  // cleans to ""), then render per verbose. annotation is last resort.
  let chosenRaw = "";
  for (const text of [readme, helpText]) {
    if (cleanReadmeSummary(text).trim()) {
      chosenRaw = text;
      break;
    }
  }

  const summary = chosenRaw
    ? verbose
      ? chosenRaw
      : cleanReadmeSummary(chosenRaw)
    : annotation;

  const srcMeta =
    typeof workflowShow["source_metadata"] === "object" && workflowShow["source_metadata"] !== null
      ? (workflowShow["source_metadata"] as Record<string, unknown>)
      : {};

  const provenance: WorkflowGuide["provenance"] = {
    version: workflowShow["version"] ?? null,
    source: {
      trs_id: srcMeta["trs_tool_id"] ?? null,
      trs_url: srcMeta["trs_url"] ?? null,
    },
  };

  if (runModel != null) {
    const stepChanges = runModel["step_version_changes"];
    provenance.freshness = {
      has_upgrade_messages: runModel["has_upgrade_messages"] ?? null,
      step_version_changes: Array.isArray(stepChanges) ? stepChanges : [],
    };
  }

  const guide: WorkflowGuide = { summary, annotation, provenance };
  if (runModel == null) {
    guide.notes = [
      "Parameter options (e.g. reference genome) resolve at run time; call again with a history_id for resolved values.",
    ];
  }
  return guide;
}

// ---------------------------------------------------------------------------
// Datatype validation helpers (for invoke-time preflight)
// ---------------------------------------------------------------------------

/**
 * The datatypes class mapping returned by GET /api/datatypes/types_and_mapping.
 * This is the inner `datatypes_mapping` object (not the outer `DatatypesCombinedMap`).
 */
export interface DatatypesMapping {
  ext_to_class_name: Record<string, string>;
  class_to_classes: Record<string, Record<string, boolean>>;
  [k: string]: unknown;
}

/**
 * True if a dataset of `suppliedExt` is acceptable where `acceptedExts` are required.
 *
 * `acceptedExts === []` means the slot accepts any datatype. Uses Galaxy's
 * datatype class hierarchy: supplied satisfies accepted if some accepted
 * extension's class is in the supplied extension's class ancestry (so `bed`
 * satisfies `tabular`). Unknown supplied/accepted extensions are treated
 * permissively -- we only reject a *provable* mismatch.
 *
 * Faithful port of Python `subtype_satisfies`.
 */
export function subtypeSatisfies(
  suppliedExt: string,
  acceptedExts: string[],
  mapping: DatatypesMapping,
): boolean {
  if (!acceptedExts.length) return true;
  const extToClass = mapping.ext_to_class_name ?? {};
  const classToClasses = mapping.class_to_classes ?? {};
  const suppliedClass = extToClass[suppliedExt];
  if (suppliedClass == null) return true; // unknown ext -> permissive
  const ancestry = classToClasses[suppliedClass] ?? {};
  for (const accepted of acceptedExts) {
    const acceptedClass = extToClass[accepted];
    if (acceptedClass == null) return true; // unknown accepted ext -> permissive
    if (acceptedClass === suppliedClass || ancestry[acceptedClass]) return true;
  }
  return false;
}

/** Direct or map-over collection type compatibility. null/undefined required == accept any shape. */
function collectionTypeCompatible(
  supplied: string | null | undefined,
  required: string | null | undefined,
): boolean {
  if (!required || !supplied) return true;
  if (supplied === required) return true;
  // map-over: a list:paired collection can feed a 'paired' (or 'list:paired') slot
  return supplied.split(":").at(-1) === required || supplied.endsWith(":" + required);
}

function extAccepted(suppliedExt: string, slot: WorkflowSlot, mapping: DatatypesMapping): boolean {
  const acc = (slot.acceptable_extensions as string[]) ?? [];
  if (acc.length) return acc.includes(suppliedExt);
  return subtypeSatisfies(suppliedExt, slot.accepted_formats as string[], mapping);
}

// ---------------------------------------------------------------------------
// validateInputs
// ---------------------------------------------------------------------------

export interface ValidationReport {
  rejects: Array<{ step_index: number; label: string; reason: string }>;
  warnings: Array<{ step_index?: number | string; message: string }>;
}

/**
 * Three-tier preflight. `supplied` keyed by str(step_index); data entries
 * carry `ext`; collection entries carry `collection_type` and optional
 * `element_extensions`; parameter entries are scalars. Rejects are provable
 * structural/datatype mismatches; warnings are inferred/uncertain.
 *
 * Faithful port of Python `validate_inputs`.
 */
export function validateInputs(
  slots: WorkflowSlot[],
  supplied: Record<string, unknown>,
  mapping: DatatypesMapping,
): ValidationReport {
  const rejects: ValidationReport["rejects"] = [];
  const warnings: ValidationReport["warnings"] = [];

  // Detect duplicate step_index values
  const seen = new Set<string>();
  for (const s of slots) {
    const idx = String(s.step_index);
    if (seen.has(idx)) {
      warnings.push({
        step_index: idx,
        message: `Workflow has multiple input slots sharing step_index ${idx}; only one will be validated.`,
      });
    }
    seen.add(idx);
  }

  const byIndex = new Map(slots.map((s) => [String(s.step_index), s]));

  for (const [key, value] of Object.entries(supplied)) {
    const slot = byIndex.get(String(key));
    if (slot == null) {
      warnings.push({
        step_index: key,
        message: `Supplied input for step ${key} has no matching workflow input slot.`,
      });
      continue;
    }

    const itype = slot.input_type;
    const isRef =
      typeof value === "object" && value !== null && "src" in (value as Record<string, unknown>);

    if (itype === "parameter") {
      if (isRef) {
        rejects.push({
          step_index: slot.step_index,
          label: slot.label,
          reason: "Parameter input given a dataset/collection reference; expected a scalar value.",
        });
      }
      continue;
    }

    if (!isRef) {
      rejects.push({
        step_index: slot.step_index,
        label: slot.label,
        reason: `${itype} input expects a {'src','id'} reference.`,
      });
      continue;
    }

    const ref = value as Record<string, unknown>;

    if (itype === "data") {
      if (ref["src"] !== "hda") {
        rejects.push({
          step_index: slot.step_index,
          label: slot.label,
          reason: `Slot expects a single dataset (hda); got a collection (${ref["src"]}).`,
        });
        continue;
      }
      const ext = ref["ext"] as string | undefined;
      if (ext && !extAccepted(ext, slot, mapping)) {
        rejects.push({
          step_index: slot.step_index,
          label: slot.label,
          reason: `Dataset datatype '${ext}' is not accepted here (expects: ${(slot.accepted_formats as string[]).join(", ")}).`,
        });
      } else if (!ext && (slot.accepted_formats as unknown[]).length) {
        warnings.push({
          step_index: slot.step_index,
          message: `Could not determine datatype of the dataset for '${slot.label}'; slot expects ${(slot.accepted_formats as string[]).join(", ")}.`,
        });
      }
    } else if (itype === "data_collection") {
      if (ref["src"] !== "hdca") {
        rejects.push({
          step_index: slot.step_index,
          label: slot.label,
          reason: `Slot expects a dataset collection (hdca); got ${ref["src"]}.`,
        });
        continue;
      }
      if (!collectionTypeCompatible(ref["collection_type"] as string | undefined, slot.collection_type as string | undefined)) {
        rejects.push({
          step_index: slot.step_index,
          label: slot.label,
          reason: `Collection type '${ref["collection_type"]}' is incompatible with the slot's '${slot.collection_type}'.`,
        });
        continue;
      }
      for (const elExt of (ref["element_extensions"] as string[] | undefined) ?? []) {
        if (!extAccepted(elExt, slot, mapping)) {
          rejects.push({
            step_index: slot.step_index,
            label: slot.label,
            reason: `Collection element datatype '${elExt}' is not accepted here (expects: ${(slot.accepted_formats as string[]).join(", ")}).`,
          });
          break;
        }
      }
    }
  }

  // Warn on missing required slots (don't block -- model may add them next call)
  for (const slot of slots) {
    if (!slot.optional && !(String(slot.step_index) in supplied)) {
      warnings.push({
        step_index: slot.step_index,
        message: `Required input '${slot.label}' (step ${slot.step_index}) not supplied yet.`,
      });
    }
  }

  return { rejects, warnings };
}
