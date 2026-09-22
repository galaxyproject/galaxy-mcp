import { z } from "zod";
import { fetchIwcWorkflows, enrichWorkflowResult, type EnrichedIwcWorkflow } from "../iwc-manifest";
import type { GalaxyContext } from "../context";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

const input = {
  query: z.string().describe("Case-insensitive substring matched against workflow name, annotation, tags, or readme"),
};
type In = { query: string };

async function run(i: In, _ctx: GalaxyContext): Promise<EnrichedIwcWorkflow[]> {
  const workflows = await fetchIwcWorkflows();
  const q = i.query.toLowerCase();

  return workflows
    .filter((wf) => {
      const def = wf.definition ?? {};
      const name = (def.name ?? "").toLowerCase();
      const annotation = (def.annotation ?? "").toLowerCase();
      const tags = (def.tags ?? []).map((t) => t.toLowerCase());
      const readme = (wf.readme ?? "").toLowerCase();
      return name.includes(q) || annotation.includes(q) || tags.some((t) => t.includes(q)) || readme.includes(q);
    })
    .map((wf) => enrichWorkflowResult(wf));
}

export const searchIwcWorkflowsOp: Operation<typeof input, EnrichedIwcWorkflow[]> = {
  name: "search_iwc_workflows",
  domain: "iwc",
  summary: "Search IWC curated workflows by substring (case-insensitive) against name, annotation, tags, or readme.",
  input,
  run,
  project: (out, i) => ({ message: `${out.length} IWC workflows matching "${i.query}"` }),
};

register(searchIwcWorkflowsOp as AnyOperation);

export const searchIwcWorkflows = (i: In, ctx: GalaxyContext) => runOperation(searchIwcWorkflowsOp, i, ctx);
