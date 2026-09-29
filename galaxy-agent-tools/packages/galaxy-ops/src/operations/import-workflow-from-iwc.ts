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
  // server.py, import_workflow_from_iwc: raised inside its try as well, so wrapped.
  if (!wf) {
    throw new GalaxyNotFoundError(
      "Failed to import workflow from IWC: " +
        `Workflow with trsID '${i.trsId}' not found in IWC manifest. ` +
        "Check the trsID format and that it exists in the IWC. " +
        "You can search workflows using search_iwc_workflows() first.",
    );
  }

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
  // server.py, import_workflow_from_iwc: the TRS id that was asked for. The id
  // Galaxy gave the imported copy is in data, which is where a caller goes next.
  project: (_out, i) => ({ message: `Successfully imported workflow '${i.trsId}'` }),
  // server.py, import_workflow_from_iwc: two clients, so two shapes -- requests for the
  // manifest, bioblend for the import -- and one sentence over both.
  failure: {
    shape: (facts) => (facts.url.includes("iwc.galaxyproject.org") ? "raise-for-status" : "bioblend-write"),
    sentence: (text) => `Failed to import workflow from IWC: ${text}`,
  },
};

register(importWorkflowFromIwcOp as AnyOperation);

export const importWorkflowFromIwc = (i: In, ctx: GalaxyContext) => runOperation(importWorkflowFromIwcOp, i, ctx);
