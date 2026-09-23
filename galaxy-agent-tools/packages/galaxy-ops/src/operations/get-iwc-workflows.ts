import { fetchIwcWorkflows, type IwcWorkflow } from "../iwc-manifest";
import type { GalaxyContext } from "../context";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

const input = {};
type In = Record<string, never>;

async function run(_i: In, _ctx: GalaxyContext): Promise<IwcWorkflow[]> {
  return fetchIwcWorkflows();
}

export const getIwcWorkflowsOp: Operation<typeof input, IwcWorkflow[]> = {
  name: "get_iwc_workflows",
  domain: "iwc",
  summary: "Fetch all workflows from the IWC (Intergalactic Workflow Commission) manifest (raw, un-enriched).",
  input,
  run,
  project: (out) => ({ message: `${out.length} IWC workflows` }),
};

register(getIwcWorkflowsOp as AnyOperation);

export const getIwcWorkflows = (_i: In, ctx: GalaxyContext) => runOperation(getIwcWorkflowsOp, _i, ctx);
