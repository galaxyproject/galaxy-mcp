import { z } from "zod";
import type { GalaxyContext } from "../context";
import { legacyGet } from "../legacy";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

/** Hand-typed: Galaxy's tool panel endpoint is not in the OpenAPI bindings. */
export interface ToolPanel {
  [k: string]: unknown;
}

const input = {};
type In = Record<string, never>;

async function run(_i: In, ctx: GalaxyContext): Promise<ToolPanel> {
  return legacyGet<ToolPanel>(ctx, "/api/tools", {
    params: { query: { in_panel: true } },
  });
}

export const getToolPanelOp: Operation<typeof input, ToolPanel> = {
  name: "get_tool_panel",
  domain: "tools",
  summary: "Return the full Galaxy tool panel (nested sections with tools). Legacy endpoint.",
  input,
  run,
  project: (_out, _i) => ({ message: "Tool panel" }),
};

register(getToolPanelOp as AnyOperation);

export const getToolPanel = (i: In, ctx: GalaxyContext) => runOperation(getToolPanelOp, i, ctx);
