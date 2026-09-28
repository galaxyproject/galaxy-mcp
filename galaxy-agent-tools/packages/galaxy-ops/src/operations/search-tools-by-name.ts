import { z } from "zod";
import type { GalaxyContext } from "../context";
import { legacyGet } from "../legacy";
import { pyLower } from "../python-str";
import { paginate, shrinkPaged, validatePagination, wirePagination, type Paged } from "./pagination";
import { register, runOperation } from "./registry";
import type { AnyOperation, InputOf, Operation } from "./types";

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
  // `pyLower`, not `toLowerCase`: the other server lowercases with tables that have not
  // been told about 55 of the code points this runtime folds, and a needle and a haystack
  // cased differently on the two sides match differently.
  const needle = pyLower(i.query);
  // A 200 carrying something other than a list is Galaxy breaking its contract.
  const matches = (Array.isArray(tools) ? tools : []).filter(
    (t) =>
      pyLower(t.name ?? "").includes(needle) ||
      pyLower(t.id ?? "").includes(needle) ||
      pyLower(t.description ?? "").includes(needle),
  );
  return paginate(matches, { limit, offset, noun: "tools" });
}

export const searchToolsByNameOp: Operation<typeof input, Paged<ToolListItem>> = {
  name: "search_tools_by_name",
  domain: "tools",
  summary:
    "Search Galaxy tools by name, id, or description substring (case-insensitive), a page at a time.",
  input,
  run,
  budget: {
    rows: (out) => out.items.length,
    shrink: (out, keep) => shrinkPaged(out, keep, "tools"),
  },
  project: (out, i) => ({
    data: out.items,
    // The other server's sentence, word for word: the budget is measured on the
    // whole envelope, message included, so a different sentence cuts the page at
    // a different row. See the note on `message` in registry.ts.
    message: `Found ${out.pagination.total} tools matching '${i.query}', returning ${out.items.length}`,
    count: out.items.length,
    pagination: wirePagination(out.pagination),
  }),
};

register(searchToolsByNameOp as AnyOperation);

// A library caller may leave the paged arguments out; run() applies the same
// defaults the schema declares for the parsed surface path.
export const searchToolsByName = (i: In, ctx: GalaxyContext) =>
  runOperation(searchToolsByNameOp, i as InputOf<typeof input>, ctx);
