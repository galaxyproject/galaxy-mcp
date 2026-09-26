import { z } from "zod";
import type { GalaxyContext } from "../context";
import { legacyGet } from "../legacy";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

/** Hand-typed: Galaxy's tool-show endpoint is not in the OpenAPI bindings. */
interface ToolShowPayload {
  id: string;
  name?: string;
  version?: string;
  citations?: unknown[];
  [k: string]: unknown;
}

export interface ToolCitationsResult {
  tool_name?: string;
  tool_version?: string;
  citations: unknown[];
}

const input = {
  toolId: z.string().describe("tool id"),
};
type In = { toolId: string };

async function run(i: In, ctx: GalaxyContext): Promise<ToolCitationsResult> {
  const payload = await legacyGet<ToolShowPayload>(ctx, "/api/tools/{tool_id}", {
    params: {
      path: { tool_id: i.toolId },
      query: { io_details: false, link_details: false },
    },
  });
  return {
    tool_name: payload.name,
    tool_version: payload.version,
    citations: payload.citations ?? [],
  };
}

export const getToolCitationsOp: Operation<typeof input, ToolCitationsResult> = {
  name: "get_tool_citations",
  domain: "tools",
  result: { kind: "object", fields: ["tool_name", "tool_version", "citations"] },
  summary: "Return citations for a Galaxy tool by id.",
  input,
  run,
  project: (out, i) => ({
    message: `${out.citations.length} citation(s) for ${out.tool_name ?? i.toolId}`,
  }),
};

register(getToolCitationsOp as AnyOperation);

export const getToolCitations = (i: In, ctx: GalaxyContext) => runOperation(getToolCitationsOp, i, ctx);
