import { z } from "zod";
import type { GetJson } from "../bindings";
import type { GalaxyContext } from "../context";
import { classifyHttp } from "../errors";
import { pyGet, pyStr } from "../python-values";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

export type WorkflowDetail = GetJson<"/api/workflows/{workflow_id}">;

const input = {
  workflowId: z.string().describe("Encoded stored-workflow id"),
  version: z.number().int().nullish().describe("Specific workflow version"),
};
type In = { workflowId: string; version?: number | null };

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
  // server.py, get_workflow_details: the workflow's name through dict.get, falling
  // back to the id that was asked for when the record carries no name key.
  project: (w, i) => ({
    message: `Retrieved details for workflow '${pyStr(pyGet(w as Record<string, unknown>, "name", i.workflowId))}'`,
  }),
};

register(getWorkflowDetailsOp as AnyOperation);

export const getWorkflowDetails = (i: In, ctx: GalaxyContext) => runOperation(getWorkflowDetailsOp, i, ctx);
