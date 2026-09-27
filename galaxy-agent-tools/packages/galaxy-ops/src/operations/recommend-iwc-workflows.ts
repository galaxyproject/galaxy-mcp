import { z } from "zod";
import {
  fetchIwcWorkflows,
  enrichWorkflowResult,
  extractToolNamesFromSteps,
  type EnrichedIwcWorkflow,
} from "../iwc-manifest";
import { tokenizeForSearch, BM25Okapi } from "../bm25";
import type { GalaxyContext } from "../context";
import { validatePagination } from "./pagination";
import { register, runOperation } from "./registry";
import type { AnyOperation, InputOf, Operation } from "./types";

const DEFAULT_LIMIT = 5;
// Python's ceiling for this tool; a window one surface refuses the other refuses.
const MAX_LIMIT = 25;

const input = {
  intent: z.string().describe("free-text description of the analysis you want"),
  limit: z.number()
    .int()
    .default(DEFAULT_LIMIT)
    .describe(`Max recommendations to return (default ${DEFAULT_LIMIT}, max ${MAX_LIMIT})`),
};
type In = { intent: string; limit?: number };

type RecommendedWorkflow = EnrichedIwcWorkflow & { match_score: number };

/** Top-N recommendations are ranked as one result, not a page window. */
export interface Recommendations {
  items: RecommendedWorkflow[];
  pagination: {
    total: number;
    returned: number;
    limit: number;
    hasNext: false;
    trimmedForSize?: boolean;
    helperText: string;
  };
}

function recommendationSummary(
  total: number,
  returned: number,
  limit: number,
  trimmedForSize = false,
): Recommendations["pagination"] {
  // A ranking cut to fit the budget is not the same as a ranking cut to the limit
  // asked for, and the Python tool says so too rather than letting the shorter set
  // read as "these are the ones that matched".
  const why = trimmedForSize
    ? ` Matches below these were dropped to fit the output budget.`
    : ` Refine the intent to change the ranking.`;
  return {
    total,
    returned,
    limit,
    hasNext: false,
    ...(trimmedForSize ? { trimmedForSize: true } : {}),
    helperText:
      returned < total
        ? `Returning the top ${returned} of ${total} matching workflows.${why}`
        : `All ${returned} matching workflows fit in the recommendation set.`,
  };
}

async function run(i: In, _ctx: GalaxyContext): Promise<Recommendations> {
  const limit = i.limit ?? DEFAULT_LIMIT;
  validatePagination(limit, 0, { maxLimit: MAX_LIMIT, pageable: false });
  const workflows = await fetchIwcWorkflows();

  // Build corpus: name appears twice for 2x weighting
  const corpus = workflows.map((wf) => {
    const def = wf.definition ?? {};
    const steps = def.steps;
    const toolNames =
      steps !== undefined && !Array.isArray(steps) && typeof steps === "object"
        ? extractToolNamesFromSteps(steps as Record<string, unknown>)
        : [];
    const parts = [
      def.name ?? "",
      def.name ?? "", // intentional 2x weight
      def.annotation ?? "",
      (def.tags ?? []).join(" "),
      wf.readme ?? "",
      toolNames.join(" "),
    ];
    return tokenizeForSearch(parts.join(" "));
  });

  const bm25 = new BM25Okapi(corpus);
  const q = tokenizeForSearch(i.intent);
  const empty = (total: number): Recommendations => ({
    items: [],
    pagination: recommendationSummary(total, 0, limit),
  });
  if (q.length === 0) return empty(0);

  const scores = bm25.getScores(q);

  const scored: Array<[typeof workflows[number], number]> = workflows
    .map((wf, idx) => [wf, scores[idx]] as [typeof workflows[number], number])
    .filter(([, s]) => s > 0);

  scored.sort((a, b) => b[1] - a[1]);
  const top = scored.slice(0, limit);

  return {
    items: top.map(([wf, score]) => ({
      ...enrichWorkflowResult(wf),
      match_score: Math.round(score * 100) / 100,
    })),
    pagination: recommendationSummary(scored.length, top.length, limit),
  };
}

export const recommendIwcWorkflowsOp: Operation<typeof input, Recommendations> = {
  name: "recommend_iwc_workflows",
  domain: "iwc",
  summary: "Rank IWC curated workflows by relevance to a free-text intent using BM25.",
  input,
  run,
  budget: {
    rows: (out) => out.items.length,
    shrink: (out, keep) => ({
      items: out.items.slice(0, keep),
      pagination: recommendationSummary(out.pagination.total, keep, out.pagination.limit, true),
    }),
  },
  // A ranking is not a page. There is no offset to walk, so there is nothing
  // truthful to put in a pagination block and the Python tool sends none -- which
  // leaves the message as the only place a cut can be reported, and it is reported
  // there rather than dropped.
  project: (out) => ({
    data: out.items,
    // The other server's sentence, word for word; the budget is measured on it
    // too, and this is the one listing whose cut is explained in the message
    // rather than in a pagination block it does not send.
    message:
      `Found ${out.items.length} workflows matching your intent` +
      (out.pagination.trimmedForSize
        ? "; additional matches were dropped to fit the output budget"
        : ""),
    count: out.items.length,
    pagination: null,
  }),
};

register(recommendIwcWorkflowsOp as AnyOperation);

// A library caller may leave the paged arguments out; run() applies the same
// defaults the schema declares for the parsed surface path.
export const recommendIwcWorkflows = (i: In, ctx: GalaxyContext) =>
  runOperation(recommendIwcWorkflowsOp, i as InputOf<typeof input>, ctx);
