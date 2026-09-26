import { z } from "zod";
import type { GetJson } from "../bindings";
import type { GalaxyContext } from "../context";
import { classifyHttp, GalaxyValidationError } from "../errors";
import { paginate, shrinkPaged, validatePagination, type Paged } from "./pagination";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

export type Workflows = GetJson<"/api/workflows">;

const DEFAULT_LIMIT = 50;
// Python's ceiling for this tool; a window one surface refuses the other refuses.
const MAX_LIMIT = 200;

const input = {
  /**
   * Declared because the Python tool declares it, and refused because the Python tool
   * refuses it. bioblend removed the parameter in 1.1.1 and raises
   * "The workflow_id parameter has been removed, use the show_workflow() method" for any
   * value that is not None, and the Python tool hands it straight to bioblend -- so this
   * listing has never fetched by id on either surface. Fetching one workflow is
   * get_workflow_details.
   */
  workflowId: z
    .string()
    .nullish()
    .describe("Not supported: use get_workflow_details to fetch one workflow by id"),
  name: z.string().nullish().describe("Return only workflows with exactly this name"),
  published: z.boolean().default(false).describe("Only published workflows (default false)"),
  limit: z.number()
    .int()
    .default(DEFAULT_LIMIT)
    .describe(`Workflows to return per page (default ${DEFAULT_LIMIT}, max ${MAX_LIMIT})`),
  offset: z.number()
    .int()
    .default(0)
    .describe("Skip the first N workflows. Pass pagination.nextOffset for the next page."),
};
type In = { workflowId?: string | null; name?: string | null; published?: boolean; limit?: number; offset?: number };

type WorkflowItem = Workflows extends readonly (infer T)[] ? T : never;

// Galaxy's /api/workflows does take a window, but bioblend's wrapper -- the
// shape this op mirrors -- does not, and neither does the name filter below.
// One fetch and a slice cannot report a total that disagrees with the page.
async function run(i: In, ctx: GalaxyContext): Promise<Paged<WorkflowItem>> {
  if (i.workflowId != null) {
    throw new GalaxyValidationError(
      "workflow_id is not supported by this listing on either surface -- bioblend removed it " +
        `and raises for any value; call get_workflow_details with id '${i.workflowId}' instead`,
    );
  }
  const limit = i.limit ?? DEFAULT_LIMIT;
  const offset = i.offset ?? 0;
  validatePagination(limit, offset, { maxLimit: MAX_LIMIT });
  const { data, error, response } = await ctx.client.GET("/api/workflows", {
    // bioblend sends show_published only when it is true (`if published:`), so a false
    // default is an omitted parameter rather than an explicit false.
    params: { query: { show_published: i.published ? true : null } },
  });
  if (error || !data) throw classifyHttp(response.status, error);
  // A 200 carrying something other than a list is Galaxy breaking its contract.
  const all = (Array.isArray(data) ? data : []) as WorkflowItem[];
  // Filter then window, by exact name, which is bioblend's `[w for w in workflows
  // if w["name"] == name]` and the order the Python tool applies it in. A name the
  // caller supplied at all is a filter, including an empty one.
  const matches =
    i.name != null ? all.filter((w) => (w as { name?: string }).name === i.name) : all;
  return paginate(matches, { limit, offset, noun: "workflows" });
}

export const listWorkflowsOp: Operation<typeof input, Paged<WorkflowItem>> = {
  name: "list_workflows",
  domain: "workflows",
  result: { kind: "object", fields: ["items", "pagination"], paginated: true },
  summary:
    "List stored workflows (id, name), a page at a time. Optional exact-name + published filter.",
  input,
  run,
  budget: {
    rows: (out) => out.items.length,
    shrink: (out, keep) => shrinkPaged(out, keep, "workflows"),
  },
  project: (out) => ({
    message: `${out.items.length} of ${out.pagination.total} workflow(s)`,
    pagination: out.pagination,
  }),
};

register(listWorkflowsOp as AnyOperation);

export const listWorkflows = (i: In, ctx: GalaxyContext) => runOperation(listWorkflowsOp, i, ctx);
