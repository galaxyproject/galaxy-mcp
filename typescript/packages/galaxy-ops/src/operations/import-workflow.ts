import { z } from "zod";
import type { GalaxyContext } from "../context";
import { GalaxyValidationError } from "../errors";
import { coerceJsonObject, jsonObjectOrString, type JsonObjectArg } from "../json-object";
import { legacyGet, legacyPost } from "../legacy";
import type { ImportedWorkflow } from "./import-workflow-from-iwc";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

const DEFAULT_FILL_DEFAULTS = false;

const input = {
  definition: jsonObjectOrString().describe(
    "The workflow as Galaxy's native .ga dict or a Format2 (gxformat2) dict, the same as a JSON object string, or Format2 YAML text (class: GalaxyWorkflow). Embedded user-defined tools are created if the user is allowed to.",
  ),
  fillDefaults: z
    .boolean()
    .default(DEFAULT_FILL_DEFAULTS)
    .describe(
      "Have Galaxy fill each tool parameter the definition leaves unset with its default while storing it, so the stored workflow carries complete tool state; without it, running the workflow may need allow_tool_state_corrections on invoke_workflow for those parameters.",
    ),
};
type In = { definition: JsonObjectArg; fillDefaults?: boolean };

/** Galaxy's record of the new stored workflow, and the tools it uses that Galaxy does not have. */
export interface ImportWorkflowResult extends ImportedWorkflow {
  /** Null when the export could not be read to check. */
  missing_tools: string[] | null;
}

/**
 * The tools a workflow's export marks as not installed, its subworkflows included.
 *
 * server.py, _missing_tools: Galaxy's export gives each step `errors: module.get_errors()`, and a
 * tool step has one only when no version of its tool is installed (ToolModule.get_errors).
 */
function missingTools(workflow: unknown, found: string[] = []): string[] {
  const steps = (workflow as { steps?: unknown } | null)?.steps;
  if (typeof steps !== "object" || steps === null || Array.isArray(steps)) return found;
  for (const step of Object.values(steps as Record<string, unknown>)) {
    if (typeof step !== "object" || step === null) continue;
    const s = step as Record<string, unknown>;
    const tool = s["tool_id"] || s["tool_uuid"];
    if (s["type"] === "tool" && s["errors"] && typeof tool === "string" && !found.includes(tool)) found.push(tool);
    missingTools(s["subworkflow"], found);
  }
  return found;
}

async function run(i: In, ctx: GalaxyContext): Promise<ImportWorkflowResult> {
  // server.py, import_workflow: checked before its try, so not wrapped.
  let workflow: Record<string, unknown> | undefined;
  if (typeof i.definition === "string" && i.definition.includes("GalaxyWorkflow")) {
    try {
      JSON.parse(i.definition);
    } catch {
      // Galaxy's own rule for workflow text (api/workflows.py, __api_import_from_archive): not
      // JSON and naming GalaxyWorkflow is Format2 YAML, which it converts itself.
      workflow = { yaml_content: i.definition };
    }
  }
  workflow ??= coerceJsonObject(i.definition, "definition");
  if (!workflow || Object.keys(workflow).length === 0) {
    throw new GalaxyValidationError("definition must be a workflow object (a .ga or Format2 dict)");
  }
  const imported = await legacyPost<ImportedWorkflow>(ctx, "/api/workflows", {
    body: { workflow, fill_defaults: i.fillDefaults ?? DEFAULT_FILL_DEFAULTS },
  });
  // Galaxy stores a workflow whose tools it lacks and leaves them out of its answer
  // (managers/workflows.py, missing_tool_tups), so they are read back off the export. The
  // workflow is stored by now: a failure here must not read as a failed import.
  let missing: string[] | null;
  try {
    // style=ga: a plain export follows the admin's default_workflow_export_format, and a
    // Format2 one carries no step errors to read.
    const exported = await legacyGet<Record<string, unknown>>(ctx, "/api/workflows/{workflow_id}/download", {
      params: { path: { workflow_id: imported.id }, query: { style: "ga" } },
    });
    missing = "a_galaxy_workflow" in exported ? missingTools(exported) : null;
  } catch {
    missing = null;
  }
  return { ...imported, missing_tools: missing };
}

export const importWorkflowOp: Operation<typeof input, ImportWorkflowResult> = {
  name: "import_workflow",
  domain: "workflows",
  summary:
    "Import a workflow from its definition into the user's Galaxy instance. The stored workflow takes " +
    "the name the definition gives it, and it stays private. Each call creates a new stored workflow, " +
    "so a retry makes a duplicate; an embedded GalaxyUserTool step also becomes a new user-defined tool " +
    "each time, unless its tool_uuid names one the user already has. Galaxy imports a workflow whose " +
    "tools it does not have; missing_tools in the result names them (null if they could not be " +
    "checked), and invoking the workflow fails until they are installed.",
  input,
  readOnly: false,
  run,
  // server.py, import_workflow: the name Galaxy stored (None when it gave none), and what the
  // export said about its tools.
  project: (out) => {
    const name = `Imported workflow '${out.name ?? "None"}'`;
    const missing = out.missing_tools;
    if (missing === null) return { message: `${name}; its tools could not be checked` };
    if (missing.length === 0) return { message: name };
    return {
      message:
        `${name}, but this Galaxy does not have ${missing.length} of its tools ` +
        `(${missing.join(", ")}); invoking it fails until they are installed`,
    };
  },
  failure: {
    shape: "bioblend-write",
    sentence: (text) => `Failed to import workflow: ${text}`,
  },
};

register(importWorkflowOp as AnyOperation);

export const importWorkflow = (i: In, ctx: GalaxyContext) =>
  runOperation(importWorkflowOp, { ...i, fillDefaults: i.fillDefaults ?? DEFAULT_FILL_DEFAULTS }, ctx);
