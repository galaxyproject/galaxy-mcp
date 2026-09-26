import { z } from "zod";
import type { GalaxyContext } from "../context";
import { GalaxyNotFoundError } from "../errors";
import { legacyGet } from "../legacy";
import { paginate, paginationInfo, validatePagination, type PaginationInfo } from "./pagination";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

interface PanelNode {
  id?: string;
  name?: string;
  description?: string;
  versions?: string[];
  model_class?: string;
  elems?: PanelNode[];
  [k: string]: unknown;
}

export interface PanelEntry {
  id: string;
  name: string;
  type: "section" | "tool";
  /** Runnable tools in the section. Absent on a bare tool. */
  tool_count?: number;
  /** Present on a bare tool sitting outside any section. */
  description?: string;
}

export interface SlimTool {
  id: string;
  name: string;
  description: string;
  versions: string[];
}

/** Counted over the whole panel, not the page. */
export interface PanelTotals {
  tool_count: number;
  section_count: number;
}

export type ToolPanelOverview = { entries: PanelEntry[]; pagination: PaginationInfo } & PanelTotals;
export type ToolPanelSection = {
  section_id: string;
  section_name: string;
  tools: SlimTool[];
  pagination: PaginationInfo;
} & PanelTotals;
export type ToolPanelResult = ToolPanelOverview | ToolPanelSection;

const DEFAULT_LIMIT = 100;
// Python's ceiling for this tool; a window one surface refuses the other refuses.
const MAX_LIMIT = 500;

const input = {
  // Null is the overview, as it is on the other surface: Python declares
  // `section_id: str | None`, and null there means "no section asked for".
  sectionId: z
    .string()
    .nullish()
    .describe("List one section's tools instead of the sections. Call with no arguments for valid ids."),
  limit: z.number()
    .int()
    .default(DEFAULT_LIMIT)
    .describe(
      `Rows per page (default ${DEFAULT_LIMIT}, max ${MAX_LIMIT}). Applies to the sections when ` +
        "sectionId is absent, and to a section's tools when it is present.",
    ),
  offset: z.number()
    .int()
    .default(0)
    .describe(
      "Skip the first N rows of whichever listing this call selects. An offset from the " +
        "overview does not carry over to a sectionId call -- start that one at 0.",
    ),
};
type In = { sectionId?: string | null; limit?: number; offset?: number };

/**
 * What the panel's three kinds of node are, copied from the Python server's
 * `_is_panel_tool` and `_summarize_panel_entry` rather than reasoned out again.
 *
 * A node is a section when it carries an `elems` key at all -- not when it says
 * ToolSection, and whatever is under the key -- and a tool when it does not and
 * Galaxy has not called it a ToolSectionLabel. There is deliberately no fallback
 * on an id ending in `_label`: the other surface has none, and a rule only one
 * side applies is a tool one side can run and the other cannot find.
 */
const isSection = (n: PanelNode): boolean => "elems" in n;
const isPanelTool = (n: PanelNode): boolean => !isSection(n) && n.model_class !== "ToolSectionLabel";

/** Walks nested sections: a page's entry count is not the server's tool count. */
function totals(nodes: PanelNode[]): PanelTotals {
  let tool_count = 0;
  let section_count = 0;
  for (const node of nodes) {
    if (isSection(node)) {
      section_count += 1;
      const inner = totals(Array.isArray(node.elems) ? node.elems : []);
      tool_count += inner.tool_count;
      section_count += inner.section_count;
    } else if (isPanelTool(node)) {
      tool_count += 1;
    }
  }
  return { tool_count, section_count };
}

const slim = (n: PanelNode): SlimTool => ({
  id: n.id ?? "",
  name: n.name ?? "",
  description: n.description ?? "",
  versions: Array.isArray(n.versions) ? n.versions : [],
});

async function run(i: In, ctx: GalaxyContext): Promise<ToolPanelResult> {
  const limit = i.limit ?? DEFAULT_LIMIT;
  const offset = i.offset ?? 0;
  validatePagination(limit, offset, { maxLimit: MAX_LIMIT });

  const panel = await legacyGet<PanelNode[]>(ctx, "/api/tools", {
    params: { query: { in_panel: true } },
  });
  const nodes = Array.isArray(panel) ? panel : [];
  const counted = totals(nodes);

  if (i.sectionId != null) {
    const section = nodes.find((n) => isSection(n) && n.id === i.sectionId);
    if (!section) {
      throw new GalaxyNotFoundError(
        `no tool panel section with id '${i.sectionId}'; ` +
          "call get_tool_panel with no arguments and use the id of an entry whose type is " +
          "'section' (an entry of type 'tool' is a tool, not a section)",
      );
    }
    const tools = (Array.isArray(section.elems) ? section.elems : []).filter(isPanelTool).map(slim);
    const page = paginate(tools, { limit, offset, noun: "tools" });
    return {
      ...counted,
      section_id: section.id ?? "",
      section_name: section.name ?? "",
      tools: page.items,
      pagination: page.pagination,
    };
  }

  // A flat limit over a nested tree would give "the first N sections, tools and
  // all", which is neither a usable overview nor a usable listing. Sections with
  // their counts, and any tool that sits outside one.
  const entries: PanelEntry[] = nodes
    .filter((n) => isSection(n) || isPanelTool(n))
    .map((n) =>
      isSection(n)
        ? {
            id: n.id ?? "",
            name: n.name ?? "",
            type: "section" as const,
            tool_count: (Array.isArray(n.elems) ? n.elems : []).filter(isPanelTool).length,
          }
        : { id: n.id ?? "", name: n.name ?? "", type: "tool" as const, description: n.description ?? "" },
    );
  const page = paginate(entries, { limit, offset, noun: "entries" });
  return { ...counted, entries: page.items, pagination: page.pagination };
}

/** The two halves of `budget.shrink`, one per shape, so neither has to know the other's key. */
const shrinkTools = (out: ToolPanelSection, keep: number) => ({
  tools: out.tools.slice(0, keep),
  pagination: reshape(out.pagination, keep, "tools"),
});
const shrinkEntries = (out: ToolPanelOverview, keep: number) => ({
  entries: out.entries.slice(0, keep),
  pagination: reshape(out.pagination, keep, "entries"),
});
const reshape = (was: PaginationInfo, keep: number, noun: string): PaginationInfo =>
  paginationInfo({
    total: was.total,
    returned: keep,
    limit: was.limit,
    offset: was.offset,
    noun,
    trimmedForSize: true,
  });

export const getToolPanelOp: Operation<typeof input, ToolPanelResult> = {
  name: "get_tool_panel",
  domain: "tools",
  result: { kind: "object", fields: ["tool_count", "section_count"], paginated: true },
  summary:
    "List the Galaxy tool panel's sections and their tool counts. Every answer carries " +
    "tool_count, how many tools the server has installed, and section_count. Pass sectionId to " +
    "list one section's tools instead. Legacy endpoint.",
  input,
  run,
  // Two shapes, so two nouns: the overview pages sections, a drill-in pages tools.
  budget: {
    rows: (out) => ("tools" in out ? out.tools.length : out.entries.length),
    shrink: (out, keep) =>
      "tools" in out
        ? { ...out, ...shrinkTools(out, keep) }
        : { ...out, ...shrinkEntries(out, keep) },
  },
  project: (out) =>
    "entries" in out
      ? { message: `${out.entries.length} of ${out.pagination.total} tool panel entries`, pagination: out.pagination }
      : {
          message: `${out.tools.length} of ${out.pagination.total} tools in ${out.section_name}`,
          pagination: out.pagination,
        },
};

register(getToolPanelOp as AnyOperation);

export const getToolPanel = (i: In, ctx: GalaxyContext) => runOperation(getToolPanelOp, i, ctx);
