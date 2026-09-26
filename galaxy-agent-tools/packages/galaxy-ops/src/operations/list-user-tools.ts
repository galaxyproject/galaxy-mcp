import { z } from "zod";
import type { GalaxyContext } from "../context";
import { legacyGet } from "../legacy";
import {
  paginate,
  shrinkPaged,
  validatePagination,
  wirePagination,
  withNoun,
  type Paged,
} from "./pagination";
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
  // "tools" is the library's noun and stays the library's noun -- the wire says
  // "user tools" because the Python tool does, and that happens in the
  // projection, where changing a sentence cannot change what run() returns.
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
    data: out.items,
    // The other server's sentence, word for word; the budget is measured on it too.
    message: `Found ${out.pagination.total} user-defined tool(s), returning ${out.items.length}`,
    count: out.items.length,
    // The other server's noun, on the wire only.
    pagination: wirePagination(withNoun(out.pagination, "user tools")),
  }),
  // server.py, list_user_tools: a raw GET whose status it checks itself, so requests' own
  // text is what the sentence quotes. format_error with two arguments -- no context.
  failure: { shape: "raise-for-status", action: "List user tools" },
};

register(listUserToolsOp as AnyOperation);

export const listUserTools = (i: In, ctx: GalaxyContext) => runOperation(listUserToolsOp, i, ctx);
