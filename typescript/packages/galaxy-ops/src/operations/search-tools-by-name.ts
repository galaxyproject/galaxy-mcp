import { z } from "zod";
import type { GalaxyContext } from "../context";
import { legacyGet } from "../legacy";
import { isPyWordChar, pyContains, pyLower, pySplitWhitespace } from "../python-str";
import { paginate, shrinkPaged, validatePagination, wirePagination, type Paged } from "./pagination";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

/** Hand-typed: Galaxy's tool list endpoint is not in the OpenAPI bindings. */
export interface ToolListItem {
  id: string;
  name?: string;
  description?: string;
  [k: string]: unknown;
}

const DEFAULT_LIMIT = 25;
// Python's ceiling for this tool; a window one surface refuses the other refuses.
const MAX_LIMIT = 100;

const input = {
  query: z.string().describe("substring matched against tool name, id, or description"),
  limit: z.number()
    .int()
    .default(DEFAULT_LIMIT)
    .describe(`Matches to return per page (default ${DEFAULT_LIMIT}, max ${MAX_LIMIT})`),
  offset: z.number()
    .int()
    .default(0)
    .describe("Skip the first N matches. Pass pagination.nextOffset for the next page."),
};
type In = { query: string; limit?: number; offset?: number };

async function run(i: In, ctx: GalaxyContext): Promise<Paged<ToolListItem>> {
  const limit = i.limit ?? DEFAULT_LIMIT;
  const offset = i.offset ?? 0;
  validatePagination(limit, offset, { maxLimit: MAX_LIMIT });
  const tools = await legacyGet<ToolListItem[]>(ctx, "/api/tools", {
    params: { query: { in_panel: false } },
  });
  // `pyLower`, not `toLowerCase`: the other server lowercases out of an older edition of
  // Unicode, both the mapping and the rule that decides whether a sigma is final, and a
  // needle and a haystack cased differently on the two sides match differently.
  const needle = pyLower(i.query);
  // A 200 carrying something other than a list is Galaxy breaking its contract.
  // `pyContains`, not `includes`, for the same reason: `in` over there compares code
  // points and `includes` compares code units, so half a surrogate pair is a substring
  // of an astral character here and never there.
  const matches = (Array.isArray(tools) ? tools : []).filter(
    (t) =>
      pyContains(pyLower(t.name ?? ""), needle) ||
      pyContains(pyLower(t.id ?? ""), needle) ||
      pyContains(pyLower(t.description ?? ""), needle),
  );
  matches.push(...wordMatches(needle, Array.isArray(tools) ? tools : [], matches));
  return paginate(matches, { limit, offset, noun: "tools" });
}

/**
 * The tools whose name, id and description hold every word of the query between them, beyond
 * those already found. A phrase need not be a substring of any one field: "count occurrences"
 * finds Count, described as "occurrences of each record". Punctuation around a word is ignored. A
 * one-word query is left to the substring match, and a one-character word, which nearly every tool
 * holds, makes the query find no more.
 */
function wordMatches(needle: string, tools: ToolListItem[], found: ToolListItem[]): ToolListItem[] {
  const words = pySplitWhitespace(needle).map(trimWord).filter(Boolean);
  if (words.length < 2 || words.some((word) => [...word].length < 2)) return [];
  const seen = new Set(found.map((t) => t.id));
  return tools.filter((t) => {
    if (seen.has(t.id)) return false;
    const fields = [t.name, t.id, t.description].map((field) => pyLower(field ?? ""));
    return words.every((word) => fields.some((field) => pyContains(field, word)));
  });
}

/** The other server's `re.sub(r"^\W+|\W+$", "", word)`: what is no word character, off both ends. */
function trimWord(word: string): string {
  const points = Array.from(word, (c) => c.codePointAt(0) ?? 0);
  let start = 0;
  let end = points.length;
  while (start < end && !isPyWordChar(points[start] ?? 0)) start++;
  while (end > start && !isPyWordChar(points[end - 1] ?? 0)) end--;
  return String.fromCodePoint(...points.slice(start, end));
}

export const searchToolsByNameOp: Operation<typeof input, Paged<ToolListItem>> = {
  name: "search_tools_by_name",
  domain: "tools",
  result: { kind: "list", paginated: true },
  summary:
    "Search Galaxy tools by name, id, or description substring (case-insensitive), then by every word of the query, a page at a time.",
  input,
  run,
  budget: {
    rows: (out) => out.items.length,
    shrink: (out, keep) => shrinkPaged(out, keep, "tools"),
  },
  project: (out, i) => ({
    data: out.items,
    // The other server's sentence, word for word: the budget is measured on the
    // whole envelope, message included, so a different sentence cuts the page at
    // a different row. See the note on `message` in registry.ts.
    message: `Found ${out.pagination.total} tools matching '${i.query}', returning ${out.items.length}`,
    count: out.items.length,
    pagination: wirePagination(out.pagination),
  }),
  // server.py, search_tools_by_name: format_error over a bioblend GET of the toolbox.
  failure: { shape: "bioblend-get", action: "Search tools", context: (i) => ({ query: i.query }) },
};

register(searchToolsByNameOp as AnyOperation);

export const searchToolsByName = (i: In, ctx: GalaxyContext) => runOperation(searchToolsByNameOp, i, ctx);
