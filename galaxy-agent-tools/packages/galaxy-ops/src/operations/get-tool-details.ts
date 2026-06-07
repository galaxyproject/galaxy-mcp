import { z } from "zod";
import type { GalaxyContext } from "../context";
import { legacyGet } from "../legacy";
import { register } from "./registry";
import type { AnyOperation, Operation } from "./types";

/** Hand-typed: Galaxy's classic tool API is not in the OpenAPI bindings (see legacy.ts). */
export interface ToolDetail {
  id: string;
  name: string;
  version?: string;
  description?: string;
  inputs?: unknown[];
  [extra: string]: unknown;
}

const input = {
  toolId: z.string().describe("Tool id, e.g. 'cat1' or 'toolshed.../fastqc/0.74'"),
  ioDetails: z.boolean().optional().describe("Include full input/output details"),
};
type In = { toolId: string; ioDetails?: boolean };

async function run(i: In, ctx: GalaxyContext): Promise<ToolDetail> {
  return legacyGet<ToolDetail>(ctx, "/api/tools/{tool_id}", {
    params: { path: { tool_id: i.toolId }, query: { io_details: i.ioDetails ?? false } },
  });
}

export const getToolDetailsOp: Operation<typeof input, ToolDetail> = {
  name: "get_tool_details",
  domain: "tools",
  summary: "Show a Galaxy tool's metadata by id (name, version, description). Legacy endpoint.",
  input,
  run,
  project: (t) => ({ message: `Tool ${t.id} (${t.name}${t.version ? " v" + t.version : ""})` }),
};

register(getToolDetailsOp as AnyOperation);

export const getToolDetails = (i: In, ctx: GalaxyContext) => getToolDetailsOp.run(i, ctx);
