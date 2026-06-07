import { z } from "zod";
import type { GetJson } from "../bindings";
import type { GalaxyContext } from "../context";
import { classifyHttp } from "../errors";
import { register } from "./registry";
import type { AnyOperation, Operation } from "./types";

export type Workflows = GetJson<"/api/workflows">;

const input = {
  name: z.string().optional().describe("Case-insensitive substring filter on workflow name"),
  published: z.boolean().optional().describe("Only published workflows"),
};
type In = { name?: string; published?: boolean };

async function run(i: In, ctx: GalaxyContext): Promise<Workflows> {
  const { data, error, response } = await ctx.client.GET("/api/workflows", {
    params: { query: { show_published: i.published ?? null } },
  });
  if (error || !data) throw classifyHttp(response.status, error);
  if (!i.name) return data;
  const needle = i.name.toLowerCase();
  return (data as Array<{ name?: string }>).filter((w) => (w.name ?? "").toLowerCase().includes(needle)) as Workflows;
}

export const listWorkflowsOp: Operation<typeof input, Workflows> = {
  name: "list_workflows",
  domain: "workflows",
  summary: "List stored workflows (id, name). Optional name substring + published filter.",
  input,
  run,
  project: (ws) => ({ message: `${(ws as unknown[]).length} workflow(s)` }),
};

register(listWorkflowsOp as AnyOperation);

export const listWorkflows = (i: In, ctx: GalaxyContext) => listWorkflowsOp.run(i, ctx);
