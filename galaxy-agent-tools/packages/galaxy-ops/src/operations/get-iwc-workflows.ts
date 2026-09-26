import { z } from "zod";
import { fetchIwcWorkflows, enrichWorkflowResult, type EnrichedIwcWorkflow } from "../iwc-manifest";
import type { GalaxyContext } from "../context";
import { paginate, shrinkPaged, validatePagination, type Paged } from "./pagination";
import { register, runOperation } from "./registry";
import type { AnyOperation, InputOf, Operation } from "./types";

const DEFAULT_LIMIT = 20;
// Python's ceiling for this tool; a window one surface refuses the other refuses.
const MAX_LIMIT = 100;

const input = {
  limit: z.number()
    .int()
    .default(DEFAULT_LIMIT)
    .describe(`Workflows to return per page (default ${DEFAULT_LIMIT}, max ${MAX_LIMIT})`),
  offset: z.number()
    .int()
    .default(0)
    .describe("Skip the first N workflows. Pass pagination.nextOffset for the next page."),
};
type In = { limit?: number; offset?: number };

async function run(i: In, _ctx: GalaxyContext): Promise<Paged<EnrichedIwcWorkflow>> {
  const limit = i.limit ?? DEFAULT_LIMIT;
  const offset = i.offset ?? 0;
  validatePagination(limit, offset, { maxLimit: MAX_LIMIT });
  const workflows = await fetchIwcWorkflows();
  // Summaries, not manifest entries: a raw entry carries its whole workflow
  // definition and runs to hundreds of KB on its own, so even one of them can
  // overrun a client. get_iwc_workflow_details is where the full record lives.
  return paginate(workflows.map((wf) => enrichWorkflowResult(wf)), { limit, offset, noun: "workflows" });
}

export const getIwcWorkflowsOp: Operation<typeof input, Paged<EnrichedIwcWorkflow>> = {
  name: "get_iwc_workflows",
  domain: "iwc",
  summary:
    "Browse IWC (Intergalactic Workflow Commission) curated workflows, a page of summaries at a time. " +
    "Call get_iwc_workflow_details for one workflow's full record.",
  input,
  run,
  budget: {
    rows: (out) => out.items.length,
    shrink: (out, keep) => shrinkPaged(out, keep, "workflows"),
  },
  project: (out) => ({
    message: `${out.items.length} of ${out.pagination.total} IWC workflows`,
    pagination: out.pagination,
  }),
};

register(getIwcWorkflowsOp as AnyOperation);

// A library caller may leave the paged arguments out; run() applies the same
// defaults the schema declares for the parsed surface path.
export const getIwcWorkflows = (i: In, ctx: GalaxyContext) =>
  runOperation(getIwcWorkflowsOp, i as InputOf<typeof input>, ctx);
