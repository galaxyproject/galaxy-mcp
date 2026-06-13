import { z } from "zod";
import type { components } from "../bindings";
import type { GalaxyContext } from "../context";
import { classifyHttp } from "../errors";
import { register } from "./registry";
import type { AnyOperation, Operation } from "./types";

export type InvocationDetail = components["schemas"]["WorkflowInvocationElementView"];

// v1 supports the detail-by-id read (invocation_id provided). The external
// get_invocations tool also lists/filters (workflow_id, history_id, view, ...) -- v2.
const input = { invocationId: z.string().describe("Encoded workflow invocation id") };

type In = { invocationId: string };

async function run(i: In, ctx: GalaxyContext): Promise<InvocationDetail> {
  const { data, error, response } = await ctx.client.GET("/api/invocations/{invocation_id}", {
    params: { path: { invocation_id: i.invocationId } },
  });
  if (error || !data) throw classifyHttp(response.status, error);
  return data as InvocationDetail;
}

export const getInvocationsOp: Operation<typeof input, InvocationDetail> = {
  name: "get_invocations", // parity: mcp-server-galaxy-py get_invocations (invocation_id -> detail)
  domain: "invocations",
  summary: "View a workflow invocation by id (id, state, steps).",
  input,
  run,
  project: (inv) => ({
    message: `Invocation ${(inv as { id?: string }).id} state=${(inv as { state?: string }).state}`,
  }),
};

register(getInvocationsOp as AnyOperation);

export const getInvocations = (i: In, ctx: GalaxyContext) => getInvocationsOp.run(i, ctx);
