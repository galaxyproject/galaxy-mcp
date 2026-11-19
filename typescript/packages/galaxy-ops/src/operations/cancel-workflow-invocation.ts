import { z } from "zod";
import type { GalaxyContext } from "../context";
import { httpError } from "../errors";
import { historyIdOf, recordHandles } from "../scope";
import { isEncodedId } from "./encoded-id";
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
  if (error || !data) throw httpError(response, error);
  recordHandles(ctx, "invocation", [(data as { id?: unknown }).id]);
  return { cancelled: true, invocation: data as Record<string, unknown> };
}

export const cancelWorkflowInvocationOp: Operation<typeof input, CancelledInvocation> = {
  name: "cancel_workflow_invocation",
  domain: "invocations",
  summary: "Cancel a running workflow invocation by id.",
  input,
  ids: { invocationId: "invocation" },
  // Cancelling cannot be undone, so under a scope the invocation is read first and the DELETE
  // never sent when it belongs to a history the run may not change.
  scope: {
    ownerOf: async (i, ctx) => {
      if (!isEncodedId(i.invocationId)) return undefined;
      const { data } = await ctx.client.GET("/api/invocations/{invocation_id}", {
        params: { path: { invocation_id: i.invocationId } },
      });
      return historyIdOf(data);
    },
    owner: (out) => historyIdOf(out.invocation),
  },
  readOnly: false,
  destructive: true,
  run,
  project: (_data, i) => ({ message: `Cancelled workflow invocation '${i.invocationId}'` }),
  // server.py, cancel_workflow_invocation.
  failure: {
    shape: "bioblend-write",
    action: "Cancel workflow invocation",
    context: (i) => ({ invocation_id: i.invocationId }),
  },
};

register(cancelWorkflowInvocationOp as AnyOperation);

export const cancelWorkflowInvocation = (i: In, ctx: GalaxyContext) => runOperation(cancelWorkflowInvocationOp, i, ctx);
