import { z } from "zod";
import type { components } from "../bindings";
import type { GalaxyContext } from "../context";
import { classifyHttp } from "../errors";
import { register } from "./registry";
import type { AnyOperation, Operation } from "./types";

export type InvocationDetail = components["schemas"]["WorkflowInvocationElementView"];

const input = { invocationId: z.string().describe("Encoded workflow invocation id") };

type In = { invocationId: string };

async function run(i: In, ctx: GalaxyContext): Promise<InvocationDetail> {
  const { data, error, response } = await ctx.client.GET("/api/invocations/{invocation_id}", {
    params: { path: { invocation_id: i.invocationId } },
  });
  if (error || !data) throw classifyHttp(response.status, error);
  return data as InvocationDetail;
}

export const getInvocationDetailsOp: Operation<typeof input, InvocationDetail> = {
  name: "get_invocation_details", // parity: /api/mcp get_invocation_details
  domain: "invocations",
  summary: "Fetch a workflow invocation's details (id, state, steps).",
  input,
  run,
  project: (inv) => ({
    message: `Invocation ${(inv as { id?: string }).id} state=${(inv as { state?: string }).state}`,
  }),
};

register(getInvocationDetailsOp as AnyOperation);

export const getInvocationDetails = (i: In, ctx: GalaxyContext) => getInvocationDetailsOp.run(i, ctx);
