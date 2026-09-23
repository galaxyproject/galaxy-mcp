import { z } from "zod";
import {
  fetchIwcWorkflows,
  enrichWorkflowResult,
  extractToolNamesFromSteps,
  type EnrichedIwcWorkflow,
} from "../iwc-manifest";
import { tokenizeForSearch, BM25Okapi } from "../bm25";
import type { GalaxyContext } from "../context";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

const input = {
  intent: z.string().describe("free-text description of the analysis you want"),
  limit: z.coerce.number().int().positive().optional().describe("max results, default 5"),
};
type In = { intent: string; limit?: number };

type RecommendedWorkflow = EnrichedIwcWorkflow & { match_score: number };

async function run(i: In, _ctx: GalaxyContext): Promise<RecommendedWorkflow[]> {
  const workflows = await fetchIwcWorkflows();
  const limit = i.limit ?? 5;

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
  if (q.length === 0) return [];

  const scores = bm25.getScores(q);

  const scored: Array<[typeof workflows[number], number]> = workflows
    .map((wf, idx) => [wf, scores[idx]] as [typeof workflows[number], number])
    .filter(([, s]) => s > 0);

  scored.sort((a, b) => b[1] - a[1]);
  const top = scored.slice(0, limit);

  return top.map(([wf, score]) => ({
    ...enrichWorkflowResult(wf),
    match_score: Math.round(score * 100) / 100,
  }));
}

export const recommendIwcWorkflowsOp: Operation<typeof input, RecommendedWorkflow[]> = {
  name: "recommend_iwc_workflows",
  domain: "iwc",
  summary: "Rank IWC curated workflows by relevance to a free-text intent using BM25.",
  input,
  run,
  project: (out, i) => ({ message: `${out.length} recommended workflow(s) for "${i.intent}"` }),
};

register(recommendIwcWorkflowsOp as AnyOperation);

export const recommendIwcWorkflows = (i: In, ctx: GalaxyContext) => runOperation(recommendIwcWorkflowsOp, i, ctx);
