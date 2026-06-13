import { z } from "zod";
import type { GetJson } from "../bindings";
import type { GalaxyContext } from "../context";
import { classifyHttp } from "../errors";
import { register } from "./registry";
import type { AnyOperation, Operation } from "./types";

export type WorkflowDetail = GetJson<"/api/workflows/{workflow_id}">;

const input = {
  workflowId: z.string().describe("Encoded stored-workflow id"),
  version: z.coerce.number().int().min(0).optional().describe("Specific workflow version"),
};
type In = { workflowId: string; version?: number };

async function run(i: In, ctx: GalaxyContext): Promise<WorkflowDetail> {
  const { data, error, response } = await ctx.client.GET("/api/workflows/{workflow_id}", {
    params: { path: { workflow_id: i.workflowId }, query: { version: i.version ?? null } },
  });
  if (error || !data) throw classifyHttp(response.status, error);
  return data as WorkflowDetail;
}

export const getWorkflowDetailsOp: Operation<typeof input, WorkflowDetail> = {
  name: "get_workflow_details",
  domain: "workflows",
  summary: "Show a stored workflow by id (name, steps, inputs).",
  input,
  run,
  project: (w) => ({ message: `Workflow ${(w as { id?: string }).id} (${(w as { name?: string }).name})` }),
};

register(getWorkflowDetailsOp as AnyOperation);

export const getWorkflowDetails = (i: In, ctx: GalaxyContext) => getWorkflowDetailsOp.run(i, ctx);
