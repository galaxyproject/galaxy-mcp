import { z } from "zod";
import type { GalaxyContext } from "../context";
import { classifyHttp } from "../errors";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

export interface CancelledInvocation {
  cancelled: true;
  invocation: Record<string, unknown>;
}

const input = {
  invocationId: z.string().describe("Workflow invocation id to cancel"),
};

type In = { invocationId: string };

async function run(i: In, ctx: GalaxyContext): Promise<CancelledInvocation> {
  const { data, error, response } = await ctx.client.DELETE("/api/invocations/{invocation_id}", {
    params: { path: { invocation_id: i.invocationId } },
  });
  if (error || !data) throw classifyHttp(response.status, error);
  return { cancelled: true, invocation: data as Record<string, unknown> };
}

export const cancelWorkflowInvocationOp: Operation<typeof input, CancelledInvocation> = {
  name: "cancel_workflow_invocation",
  domain: "invocations",
  summary: "Cancel a running workflow invocation by id.",
  input,
  readOnly: false,
  destructive: true,
  run,
  project: (_data, i) => ({ message: `Cancelled invocation ${i.invocationId}` }),
};

register(cancelWorkflowInvocationOp as AnyOperation);

export const cancelWorkflowInvocation = (i: In, ctx: GalaxyContext) => runOperation(cancelWorkflowInvocationOp, i, ctx);
