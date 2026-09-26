import { z } from "zod";
import type { GalaxyContext } from "../context";
import { legacyGet } from "../legacy";
import { paginate, shrinkPaged, validatePagination, type Paged } from "./pagination";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

/** Hand-typed: user-defined tool record from /api/unprivileged_tools. */
export interface UserTool {
  id?: string;
  uuid?: string;
  tool_id?: string;
  active?: boolean;
  [k: string]: unknown;
}

const DEFAULT_LIMIT = 25;
// Python's ceiling for this tool; a window one surface refuses the other refuses.
const MAX_LIMIT = 100;

const input = {
  active: z.boolean().default(true).describe("filter by active state, default true"),
  limit: z.number()
    .int()
    .default(DEFAULT_LIMIT)
    .describe(`Tools to return per page (default ${DEFAULT_LIMIT}, max ${MAX_LIMIT})`),
  offset: z.number()
    .int()
    .default(0)
    .describe("Skip the first N tools. Pass pagination.nextOffset for the next page."),
};
type In = { active?: boolean; limit?: number; offset?: number };

async function run(i: In, ctx: GalaxyContext): Promise<Paged<UserTool>> {
  const limit = i.limit ?? DEFAULT_LIMIT;
  const offset = i.offset ?? 0;
  validatePagination(limit, offset, { maxLimit: MAX_LIMIT });
  const tools = await legacyGet<UserTool[]>(ctx, "/api/unprivileged_tools", {
    params: { query: { active: i.active ?? true } },
  });
  // A 200 carrying something other than a list is Galaxy breaking its contract;
  // an empty page says so without crashing the whole MCP handler.
  return paginate(Array.isArray(tools) ? tools : [], { limit, offset, noun: "tools" });
}

export const listUserToolsOp: Operation<typeof input, Paged<UserTool>> = {
  name: "list_user_tools",
  domain: "userTools",
  summary: "List user-defined tools belonging to the current user, a page at a time.",
  input,
  run,
  budget: {
    rows: (out) => out.items.length,
    shrink: (out, keep) => shrinkPaged(out, keep, "tools"),
  },
  project: (out) => ({
    message: `${out.items.length} of ${out.pagination.total} user-defined tool(s)`,
    pagination: out.pagination,
  }),
};

register(listUserToolsOp as AnyOperation);

export const listUserTools = (i: In, ctx: GalaxyContext) => runOperation(listUserToolsOp, i, ctx);
