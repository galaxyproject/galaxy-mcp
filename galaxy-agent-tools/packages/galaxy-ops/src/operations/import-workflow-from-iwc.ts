import { z } from "zod";
import { fetchIwcWorkflows } from "../iwc-manifest";
import type { GalaxyContext } from "../context";
import { GalaxyNotFoundError } from "../errors";
import { legacyPost } from "../legacy";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

/** Hand-typed: the /api/workflows import endpoint returns a stored-workflow summary. */
export interface ImportedWorkflow {
  id: string;
  name?: string;
  [k: string]: unknown;
}

const input = {
  trsId: z.string().describe("TRS id of the IWC workflow to import"),
};
type In = { trsId: string };

async function run(i: In, ctx: GalaxyContext): Promise<ImportedWorkflow> {
  const workflows = await fetchIwcWorkflows();
  const wf = workflows.find((w) => w.trsID === i.trsId);
  if (!wf) throw new GalaxyNotFoundError(`IWC workflow ${i.trsId} not found`);

  return legacyPost<ImportedWorkflow>(ctx, "/api/workflows", {
    body: { workflow: wf.definition },
  });
}

export const importWorkflowFromIwcOp: Operation<typeof input, ImportedWorkflow> = {
  name: "import_workflow_from_iwc",
  domain: "iwc",
  summary: "Import an IWC curated workflow into the connected Galaxy instance by TRS id.",
  input,
  readOnly: false,
  run,
  project: (out) => ({
    message: `Imported workflow ${out.id}${out.name ? ` (${out.name})` : ""}`,
  }),
};

register(importWorkflowFromIwcOp as AnyOperation);

export const importWorkflowFromIwc = (i: In, ctx: GalaxyContext) => runOperation(importWorkflowFromIwcOp, i, ctx);
