import { z } from "zod";
import type { GalaxyContext } from "../context";
import { legacyGet } from "../legacy";
import { buildInputTemplate, summarizeToolInputs } from "../tool-inputs";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

/** Hand-typed: Galaxy's tool-show endpoint is not in the OpenAPI bindings. */
interface ToolInfo {
  id?: string;
  inputs?: unknown[];
  [k: string]: unknown;
}

export interface ToolInputTemplateResult {
  tool_id: string;
  inputs_template: Record<string, unknown>;
  parameters: unknown[];
}

const input = {
  toolId: z.string().describe("tool id"),
};
type In = { toolId: string };

async function run(i: In, ctx: GalaxyContext): Promise<ToolInputTemplateResult> {
  const info = await legacyGet<ToolInfo>(ctx, "/api/tools/{tool_id}", {
    params: {
      path: { tool_id: i.toolId },
      query: { io_details: true, link_details: false },
    },
  });
  const inputs = info.inputs ?? [];
  return {
    tool_id: i.toolId,
    inputs_template: buildInputTemplate(inputs),
    parameters: summarizeToolInputs(inputs),
  };
}

export const getToolInputTemplateOp: Operation<typeof input, ToolInputTemplateResult> = {
  name: "get_tool_input_template",
  domain: "tools",
  summary:
    "Return a ready-to-fill inputs skeleton for a Galaxy tool, plus a compact parameter summary. Call before run_tool when unsure how to shape inputs.",
  input,
  run,
  project: (out, i) => ({
    // server.py, get_tool_input_template: the same two sentences, backtick-quoted
    // `inputs` and all. The parameter count is not in it -- the summary is in data.
    message:
      `Built an input template for tool '${i.toolId}'. Replace placeholders ` +
      "(e.g. <dataset_id>) and pass the result as `inputs` to run_tool.",
  }),
};

register(getToolInputTemplateOp as AnyOperation);

export const getToolInputTemplate = (i: In, ctx: GalaxyContext) => runOperation(getToolInputTemplateOp, i, ctx);
