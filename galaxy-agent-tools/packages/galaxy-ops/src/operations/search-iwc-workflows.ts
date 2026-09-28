import { z } from "zod";
import { fetchIwcWorkflows, enrichWorkflowResult, type EnrichedIwcWorkflow } from "../iwc-manifest";
import type { GalaxyContext } from "../context";
import { pyLower } from "../python-str";
import { paginate, shrinkPaged, validatePagination, wirePagination, type Paged } from "./pagination";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

const DEFAULT_LIMIT = 20;
// Python's ceiling for this tool; a window one surface refuses the other refuses.
const MAX_LIMIT = 100;

const input = {
  query: z.string().describe("Case-insensitive substring matched against workflow name, annotation, tags, or readme"),
  limit: z.number()
    .int()
    .default(DEFAULT_LIMIT)
    .describe(`Matches to return per page (default ${DEFAULT_LIMIT}, max ${MAX_LIMIT})`),
  offset: z.number()
    .int()
    .default(0)
    .describe("Skip the first N matches. Pass pagination.nextOffset for the next page."),
};
type In = { query: string; limit?: number; offset?: number };

async function run(i: In, _ctx: GalaxyContext): Promise<Paged<EnrichedIwcWorkflow>> {
  const limit = i.limit ?? DEFAULT_LIMIT;
  const offset = i.offset ?? 0;
  validatePagination(limit, offset, { maxLimit: MAX_LIMIT });
  const workflows = await fetchIwcWorkflows();
  // The other server's `lower()`; this runtime's reads a newer edition of Unicode and
  // answers differently about what to fold and about where a word ends.
  const q = pyLower(i.query);

  const matches = workflows
    .filter((wf) => {
      const def = wf.definition ?? {};
      const name = pyLower(def.name ?? "");
      const annotation = pyLower(def.annotation ?? "");
      const tags = (def.tags ?? []).map((t) => pyLower(t));
      const readme = pyLower(wf.readme ?? "");
      return name.includes(q) || annotation.includes(q) || tags.some((t) => t.includes(q)) || readme.includes(q);
    })
    .map((wf) => enrichWorkflowResult(wf));

  return paginate(matches, { limit, offset, noun: "workflows" });
}

export const searchIwcWorkflowsOp: Operation<typeof input, Paged<EnrichedIwcWorkflow>> = {
  name: "search_iwc_workflows",
  domain: "iwc",
  result: { kind: "object", fields: ["items", "pagination"], paginated: true },
  summary: "Search IWC curated workflows by substring (case-insensitive) against name, annotation, tags, or readme.",
  input,
  run,
  budget: {
    rows: (out) => out.items.length,
    shrink: (out, keep) => shrinkPaged(out, keep, "workflows"),
  },
  project: (out, i) => ({
    data: out.items,
    // The other server's sentence, word for word; the budget is measured on it too.
    message:
      `Found ${out.pagination.total} IWC workflows matching '${i.query}', ` +
      `returning ${out.items.length}`,
    count: out.items.length,
    pagination: wirePagination(out.pagination),
  }),
};

register(searchIwcWorkflowsOp as AnyOperation);

export const searchIwcWorkflows = (i: In, ctx: GalaxyContext) => runOperation(searchIwcWorkflowsOp, i, ctx);
