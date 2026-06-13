import { z } from "zod";
import type { GalaxyContext } from "../context";
import { executeToolRequest, type ToolRun } from "../execute-tool-request";
import { register } from "./registry";
import type { AnyOperation, Operation } from "./types";

// Nested inputs only. We accept an opaque record (server `strict:true` is the real gate);
// the type documents the intended shape without flattening.
const input = {
  toolId: z.string().describe("Tool id, e.g. 'fastqc/0.74'"),
  historyId: z.string().describe("Encoded history id to run in"),
  inputs: z
    .record(z.string(), z.unknown())
    .describe("Nested tool inputs: data refs as {src:'hda',id}, batches as {__class__:'Batch',values:[...]}"),
  toolVersion: z.string().optional().describe("Optional explicit tool version"),
};

type RunToolInput = {
  toolId: string;
  historyId: string;
  inputs: Record<string, unknown>;
  toolVersion?: string;
};

async function run(i: RunToolInput, ctx: GalaxyContext): Promise<ToolRun> {
  return executeToolRequest(
    { toolId: i.toolId, historyId: i.historyId, inputs: i.inputs, toolVersion: i.toolVersion },
    ctx,
  );
}

export const runToolOp: Operation<typeof input, ToolRun> = {
  name: "run_tool", // parity: mcp-server-galaxy-py run_tool
  domain: "tools",
  summary:
    "Run a Galaxy tool via the typed tool-request path and wait until it reaches terminal state. " +
    "Inputs are nested (no flat pipe-keys); success is derived from the spawned jobs.",
  input,
  run,
  project: (o) => ({ message: `Tool run ${o.toolRequestId} completed (${o.jobs.length} job(s), state=${o.state})` }),
};

register(runToolOp as AnyOperation);

export const runTool = (i: RunToolInput, ctx: GalaxyContext) => runToolOp.run(i, ctx);
