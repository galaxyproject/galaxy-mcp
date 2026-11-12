import { z } from "zod";
import type { GalaxyContext } from "../context";
import { GalaxyValidationError } from "../errors";
import { jsonObject } from "../json-object";
import { legacyPost } from "../legacy";
import type { ImportedWorkflow } from "./import-workflow-from-iwc";
import { coerceJsonObject } from "./invoke-workflow";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

const input = {
  definition: z
    .union([jsonObject(), z.string()])
    .describe(
      "The workflow as Galaxy's native .ga dict or a Format2 (gxformat2) dict, or the same as a JSON object string. Embedded user-defined tools are created if the user is allowed to.",
    ),
};
type In = { definition: Record<string, unknown> | string };

async function run(i: In, ctx: GalaxyContext): Promise<ImportedWorkflow> {
  // server.py, import_workflow: checked before its try, so not wrapped.
  const workflow = coerceJsonObject(i.definition, "definition");
  if (!workflow || Object.keys(workflow).length === 0) {
    throw new GalaxyValidationError("definition must be a workflow object (a .ga or Format2 dict)");
  }
  return legacyPost<ImportedWorkflow>(ctx, "/api/workflows", { body: { workflow } });
}

export const importWorkflowOp: Operation<typeof input, ImportedWorkflow> = {
  name: "import_workflow",
  domain: "workflows",
  summary: "Import a workflow from its .ga or Format2 definition into the connected Galaxy instance.",
  input,
  readOnly: false,
  run,
  // server.py, import_workflow: the name Galaxy stored, None when it gave none.
  project: (out) => ({ message: `Imported workflow '${out.name ?? "None"}'` }),
  failure: {
    shape: "bioblend-write",
    sentence: (text) => `Failed to import workflow: ${text}`,
  },
};

register(importWorkflowOp as AnyOperation);

export const importWorkflow = (i: In, ctx: GalaxyContext) => runOperation(importWorkflowOp, i, ctx);
