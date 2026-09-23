import { z } from "zod";
import type { GalaxyContext } from "../context";
import { legacyGet } from "../legacy";
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

const input = {
  active: z.boolean().optional().describe("filter by active state, default true"),
};
type In = { active?: boolean };

async function run(i: In, ctx: GalaxyContext): Promise<UserTool[]> {
  return legacyGet<UserTool[]>(ctx, "/api/unprivileged_tools", {
    params: { query: { active: i.active ?? true } },
  });
}

export const listUserToolsOp: Operation<typeof input, UserTool[]> = {
  name: "list_user_tools",
  domain: "userTools",
  summary: "List user-defined tools belonging to the current user.",
  input,
  run,
  project: (out) => ({ message: `${out.length} user-defined tool(s)` }),
};

register(listUserToolsOp as AnyOperation);

export const listUserTools = (i: In, ctx: GalaxyContext) => runOperation(listUserToolsOp, i, ctx);
