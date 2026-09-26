import { z } from "zod";
import type { GalaxyContext } from "../context";
import { legacyGet } from "../legacy";
import { paginate, shrinkPaged, validatePagination, type Paged } from "./pagination";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

/** Hand-typed: Galaxy's tool list endpoint is not in the OpenAPI bindings. */
export interface ToolListItem {
  id: string;
  name?: string;
  description?: string;
  [k: string]: unknown;
}

const DEFAULT_LIMIT = 25;
// Python's ceiling for this tool; a window one surface refuses the other refuses.
const MAX_LIMIT = 100;

const input = {
  query: z.string().describe("substring matched against tool name, id, or description"),
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

async function run(i: In, ctx: GalaxyContext): Promise<Paged<ToolListItem>> {
  const limit = i.limit ?? DEFAULT_LIMIT;
  const offset = i.offset ?? 0;
  validatePagination(limit, offset, { maxLimit: MAX_LIMIT });
  const tools = await legacyGet<ToolListItem[]>(ctx, "/api/tools", {
    params: { query: { in_panel: false } },
  });
  const needle = i.query.toLowerCase();
  // A 200 carrying something other than a list is Galaxy breaking its contract.
  const matches = (Array.isArray(tools) ? tools : []).filter(
    (t) =>
      (t.name ?? "").toLowerCase().includes(needle) ||
      (t.id ?? "").toLowerCase().includes(needle) ||
      (t.description ?? "").toLowerCase().includes(needle),
  );
  return paginate(matches, { limit, offset, noun: "tools" });
}

export const searchToolsByNameOp: Operation<typeof input, Paged<ToolListItem>> = {
  name: "search_tools_by_name",
  domain: "tools",
  result: { kind: "object", fields: ["items", "pagination"], paginated: true },
  summary:
    "Search Galaxy tools by name, id, or description substring (case-insensitive), a page at a time.",
  input,
  run,
  budget: {
    rows: (out) => out.items.length,
    shrink: (out, keep) => shrinkPaged(out, keep, "tools"),
  },
  project: (out, i) => ({
    message: `${out.items.length} of ${out.pagination.total} tool(s) matching "${i.query}"`,
    pagination: out.pagination,
  }),
};

register(searchToolsByNameOp as AnyOperation);

export const searchToolsByName = (i: In, ctx: GalaxyContext) => runOperation(searchToolsByNameOp, i, ctx);
