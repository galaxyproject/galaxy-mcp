import { z } from "zod";
import type { GalaxyContext } from "../context";
import { legacyGet } from "../legacy";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

/** Hand-typed: Galaxy's tool list endpoint is not in the OpenAPI bindings. */
export interface ToolListItem {
  id: string;
  name?: string;
  description?: string;
  [k: string]: unknown;
}

const input = {
  query: z.string().describe("substring matched against tool name, id, or description"),
};
type In = { query: string };

async function run(i: In, ctx: GalaxyContext): Promise<ToolListItem[]> {
  const tools = await legacyGet<ToolListItem[]>(ctx, "/api/tools", {
    params: { query: { in_panel: false } },
  });
  const needle = i.query.toLowerCase();
  return tools.filter(
    (t) =>
      (t.name ?? "").toLowerCase().includes(needle) ||
      (t.id ?? "").toLowerCase().includes(needle) ||
      (t.description ?? "").toLowerCase().includes(needle),
  );
}

export const searchToolsByNameOp: Operation<typeof input, ToolListItem[]> = {
  name: "search_tools_by_name",
  domain: "tools",
  summary: "Search Galaxy tools by name, id, or description substring (case-insensitive).",
  input,
  run,
  project: (out, i) => ({ message: `${out.length} tool(s) matching "${i.query}"` }),
};

register(searchToolsByNameOp as AnyOperation);

export const searchToolsByName = (i: In, ctx: GalaxyContext) => runOperation(searchToolsByNameOp, i, ctx);
