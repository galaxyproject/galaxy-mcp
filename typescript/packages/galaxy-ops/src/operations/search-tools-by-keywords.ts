import { z } from "zod";
import type { GalaxyContext } from "../context";
import { legacyGet } from "../legacy";
import { pyContains, pyLower } from "../python-str";
import { paginate, shrinkPaged, validatePagination, wirePagination, type Paged } from "./pagination";
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
  model_class?: string;
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

const DEFAULT_LIMIT = 50;
// Python's ceiling for this tool; a window one surface refuses the other refuses.
const MAX_LIMIT = 200;

const input = {
  keywords: z.array(z.string()).describe("substring keywords matched against tool name/description/input extensions"),
  limit: z.number()
    .int()
    .default(DEFAULT_LIMIT)
    .describe(`Matches to return per page (default ${DEFAULT_LIMIT}, max ${MAX_LIMIT})`),
  offset: z.number()
    .int()
    .default(0)
    .describe("Skip the first N matches. Pass pagination.nextOffset for the next page."),
};
type In = { keywords: string[]; limit?: number; offset?: number };

async function run(i: In, ctx: GalaxyContext): Promise<Paged<ToolKeywordMatch>> {
  const limit = i.limit ?? DEFAULT_LIMIT;
  const offset = i.offset ?? 0;
  validatePagination(limit, offset, { maxLimit: MAX_LIMIT });
  const panel = await legacyGet<PanelNode[]>(ctx, "/api/tools", {
    params: { query: { in_panel: true } },
  });

  const allTools = flattenTools(panel).filter((t) => t.id);

  // The other server's `lower()`, mapping and final-sigma rule alike, which this runtime's
  // answers differently because it reads a newer edition of Unicode.
  const needles = i.keywords.map((k) => pyLower(k));

  const matchesImmediately = (t: PanelNode) => {
    const name = pyLower(t.name ?? "");
    const desc = pyLower(t.description ?? "");
    // `pyContains`, not `includes`: `in` compares code points over there and this
    // compares code units, which differ wherever a needle is half a surrogate pair.
    return needles.some((kw) => pyContains(name, kw) || pyContains(desc, kw));
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
      // Where the Python keyword search drops a divider: on the detail path only.
      // One that matched by name or description is already in the other list and
      // comes back as a result there -- deliberately, on both surfaces -- and one
      // that did not is not worth a request. This is that op's rule; get_tool_panel
      // copies a different Python helper and answers differently about the same node.
      if (tool.id!.endsWith("_label")) return null;
      try {
        const detail = await legacyGet<ToolDetailPayload>(ctx, "/api/tools/{tool_id}", {
          params: { path: { tool_id: tool.id! }, query: { io_details: true } },
        });
        const inputs: ToolInput[] = detail.inputs ?? [];
        const matched = inputs.some((inp) => {
          const ext = inp.extensions;
          if (Array.isArray(ext)) {
            return ext.some(
              (e) => typeof e === "string" && needles.some((kw) => pyContains(pyLower(e), kw)),
            );
          }
          if (typeof ext === "string" && ext) {
            return needles.some((kw) => pyContains(pyLower(ext), kw));
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

  // Name matches first, then extension matches, each in panel order -- mapLimit
  // writes its results by index rather than by completion, so neither group
  // reshuffles between calls. Paging a set that reordered itself would skip and
  // duplicate, which is the only property this order has to have.
  const matches = [...immediateMatches, ...extensionMatches].map((t) => ({
    id: t.id!,
    ...(t.name != null ? { name: t.name } : {}),
    ...(t.description != null ? { description: t.description } : {}),
    ...(t.versions != null ? { versions: t.versions } : {}),
  }));
  return paginate(matches, { limit, offset, noun: "tools" });
}

export const searchToolsByKeywordsOp: Operation<typeof input, Paged<ToolKeywordMatch>> = {
  name: "search_tools_by_keywords",
  domain: "tools",
  summary: "Search Galaxy tools by keywords matched against name, description, and input file extensions.",
  input,
  run,
  budget: {
    rows: (out) => out.items.length,
    shrink: (out, keep) => shrinkPaged(out, keep, "tools"),
  },
  project: (out, i) => ({
    data: out.items,
    // The other server's sentence, word for word; the budget is measured on it too.
    message:
      `Found ${out.pagination.total} tools matching keywords: ` +
      `${i.keywords.join(", ")}, returning ${out.items.length}`,
    count: out.items.length,
    pagination: wirePagination(out.pagination),
  }),
  // server.py, search_tools_by_keywords: its own sentence, not format_error's, so no hint
  // and no context. The per-tool detail fetches are best-effort on both sides.
  failure: {
    shape: "bioblend-get",
    sentence: (text) => `Failed to search tools by keywords: ${text}`,
  },
};

register(searchToolsByKeywordsOp as AnyOperation);

export const searchToolsByKeywords = (i: In, ctx: GalaxyContext) => runOperation(searchToolsByKeywordsOp, i, ctx);
