import { z } from "zod";
import type { GalaxyContext } from "../context";
import { legacyGet } from "../legacy";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

export interface ToolKeywordMatch {
  id: string;
  name?: string;
  description?: string;
  versions?: string[];
}

/** Bounded concurrency helper -- runs fn on each item with at most `limit` in flight. */
async function mapLimit<T, R>(items: T[], limit: number, fn: (t: T) => Promise<R>): Promise<R[]> {
  const results: R[] = [];
  let index = 0;

  async function worker(): Promise<void> {
    while (index < items.length) {
      const i = index++;
      results[i] = await fn(items[i]!);
    }
  }

  const workers = Array.from({ length: Math.min(limit, items.length) }, () => worker());
  await Promise.all(workers);
  return results;
}

interface PanelNode {
  id?: string;
  name?: string;
  description?: string;
  versions?: string[];
  elems?: PanelNode[];
  [k: string]: unknown;
}

interface ToolInput {
  extensions?: string | string[];
  [k: string]: unknown;
}

interface ToolDetailPayload {
  id: string;
  inputs?: ToolInput[];
  [k: string]: unknown;
}

/** Recursively flatten a panel tree into leaf tool nodes. */
function flattenTools(node: PanelNode | PanelNode[]): PanelNode[] {
  if (Array.isArray(node)) {
    return node.flatMap(flattenTools);
  }
  if (node.elems != null) {
    return node.elems.flatMap(flattenTools);
  }
  return [node];
}

const input = {
  keywords: z.array(z.string()).describe("substring keywords matched against tool name/description/input extensions"),
};
type In = { keywords: string[] };

async function run(i: In, ctx: GalaxyContext): Promise<ToolKeywordMatch[]> {
  const panel = await legacyGet<PanelNode[]>(ctx, "/api/tools", {
    params: { query: { in_panel: true } },
  });

  // Pre-filter _label ids before the immediate/detail split -- intentional deviation from the
  // Python source, which checks _label only in the detail path. Labels should never surface as results.
  const allTools = flattenTools(panel).filter((t) => t.id && !t.id.endsWith("_label"));

  const needles = i.keywords.map((k) => k.toLowerCase());

  const matchesImmediately = (t: PanelNode) => {
    const name = (t.name ?? "").toLowerCase();
    const desc = (t.description ?? "").toLowerCase();
    return needles.some((kw) => name.includes(kw) || desc.includes(kw));
  };

  const immediateMatches: PanelNode[] = [];
  const toFetch: PanelNode[] = [];
  for (const tool of allTools) {
    if (matchesImmediately(tool)) {
      immediateMatches.push(tool);
    } else {
      toFetch.push(tool);
    }
  }

  // Fetch detail for non-immediate tools with bounded concurrency.
  const extensionMatches: PanelNode[] = [];
  if (toFetch.length > 0) {
    const results = await mapLimit(toFetch, 10, async (tool) => {
      try {
        const detail = await legacyGet<ToolDetailPayload>(ctx, "/api/tools/{tool_id}", {
          params: { path: { tool_id: tool.id! }, query: { io_details: true } },
        });
        const inputs: ToolInput[] = detail.inputs ?? [];
        const matched = inputs.some((inp) => {
          const ext = inp.extensions;
          if (Array.isArray(ext)) {
            return ext.some((e) => typeof e === "string" && needles.some((kw) => e.toLowerCase().includes(kw)));
          }
          if (typeof ext === "string" && ext) {
            return needles.some((kw) => ext.toLowerCase().includes(kw));
          }
          return false;
        });
        return matched ? tool : null;
      } catch {
        return null;
      }
    });
    for (const r of results) {
      if (r != null) extensionMatches.push(r);
    }
  }

  return [...immediateMatches, ...extensionMatches].map((t) => ({
    id: t.id!,
    ...(t.name != null ? { name: t.name } : {}),
    ...(t.description != null ? { description: t.description } : {}),
    ...(t.versions != null ? { versions: t.versions } : {}),
  }));
}

export const searchToolsByKeywordsOp: Operation<typeof input, ToolKeywordMatch[]> = {
  name: "search_tools_by_keywords",
  domain: "tools",
  summary: "Search Galaxy tools by keywords matched against name, description, and input file extensions.",
  input,
  run,
  project: (out, _i) => ({ message: `${out.length} tool(s) matching keywords` }),
};

register(searchToolsByKeywordsOp as AnyOperation);

export const searchToolsByKeywords = (i: In, ctx: GalaxyContext) => runOperation(searchToolsByKeywordsOp, i, ctx);
