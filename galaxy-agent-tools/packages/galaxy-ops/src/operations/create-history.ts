import { z } from "zod";
import type { PostJson } from "../bindings";
import type { GalaxyContext } from "../context";
import { httpError } from "../errors";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

export type CreatedHistory = PostJson<"/api/histories">;

const input = { historyName: z.string().min(1).describe("Name for the new history") };
type In = { historyName: string };

async function run(i: In, ctx: GalaxyContext): Promise<CreatedHistory> {
  // The endpoint is form-urlencoded; serialize the body accordingly. The body type marks
  // all_datasets/archive_type required, but the server defaults them -- only `name` is sent,
  // so the `as never` cast is intentional (keep it; do not "fix" the required fields).
  const { data, error, response } = await ctx.client.POST("/api/histories", {
    body: { name: i.historyName } as never,
    bodySerializer: (b: unknown) => new URLSearchParams(b as Record<string, string>).toString(),
    headers: { "content-type": "application/x-www-form-urlencoded" },
  });
  if (error || !data) throw httpError(response, error);
  return data as CreatedHistory;
}

export const createHistoryOp: Operation<typeof input, CreatedHistory> = {
  name: "create_history",
  domain: "histories",
  summary: "Create a new history with the given name.",
  input,
  readOnly: false,
  run,
  // The name that was ASKED for, which is the one server.py's sentence carries
  // (create_history): Galaxy is free to hand back a different one, and the caller
  // reading this line is being told what it requested went through.
  project: (_h, i) => ({ message: `Created history '${i.historyName}'` }),
  // server.py, create_history: alone among the tools, it has no try block at all, so
  // bioblend's own text is the whole sentence -- no action, no hint, no context.
  failure: { shape: "bioblend-write" },
};

register(createHistoryOp as AnyOperation);

export const createHistory = (i: In, ctx: GalaxyContext) => runOperation(createHistoryOp, i, ctx);
