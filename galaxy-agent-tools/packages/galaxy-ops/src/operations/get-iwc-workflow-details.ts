import { z } from "zod";
import { fetchIwcWorkflows, enrichWorkflowResult, type EnrichedIwcWorkflow } from "../iwc-manifest";
import type { GalaxyContext } from "../context";
import { GalaxyNotFoundError } from "../errors";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";
import { stepsInOrder } from "../workflow-steps";

// Extend the enriched type with details-only fields
export interface IwcWorkflowDetail extends EnrichedIwcWorkflow {
  inputs: Array<{ name: string | null; type: string; annotation: string | null }>;
  outputs: Array<{ name: string | null; step: string | null }>;
  updated: string;
}

const input = {
  trsId: z.string().describe("TRS id of the IWC workflow"),
};
type In = { trsId: string };

const INPUT_TYPES = new Set(["data_input", "data_collection_input", "parameter_input"]);

// The other server reads these with dict.get(key, default): only an ABSENT key
// gets the default, a key that is present with null or "" is the value.
const orDefault = (o: Record<string, unknown>, key: string, fallback: unknown): unknown =>
  Object.prototype.hasOwnProperty.call(o, key) ? o[key] : fallback;

async function run(i: In, _ctx: GalaxyContext): Promise<IwcWorkflowDetail> {
  const workflows = await fetchIwcWorkflows();
  const wf = workflows.find((w) => w.trsID === i.trsId);
  if (!wf) throw new GalaxyNotFoundError(`IWC workflow ${i.trsId} not found`);

  const enriched = enrichWorkflowResult(wf, { fullReadme: true });

  const definition = wf.definition ?? {};
  const steps = definition.steps;

  const inputs: IwcWorkflowDetail["inputs"] = [];
  const outputs: IwcWorkflowDetail["outputs"] = [];

  if (steps && !Array.isArray(steps) && typeof steps === "object") {
    // Stated order rather than the runtime's own, for the reason spelled out on
    // stepsInOrder: the two servers' objects do not agree on one.
    for (const [stepId, stepData] of stepsInOrder(steps as Record<string, unknown>)) {
      if (!stepData || typeof stepData !== "object") continue;
      const step = stepData as Record<string, unknown>;
      const stepType = typeof step["type"] === "string" ? step["type"] : "";

      if (INPUT_TYPES.has(stepType)) {
        inputs.push({
          name: orDefault(step, "label", `Input ${stepId}`) as string | null,
          type: stepType,
          annotation: orDefault(step, "annotation", "") as string | null,
        });
      }

      const workflowOutputs = step["workflow_outputs"];
      if (Array.isArray(workflowOutputs)) {
        for (const wo of workflowOutputs) {
          if (!wo || typeof wo !== "object") continue;
          const woObj = wo as Record<string, unknown>;
          const label = orDefault(woObj, "label", orDefault(woObj, "output_name", ""));
          const stepLabel = orDefault(step, "label", `Step ${stepId}`);
          outputs.push({ name: label as string | null, step: stepLabel as string | null });
        }
      }
    }
  }

  return {
    ...enriched,
    inputs,
    outputs,
    updated: typeof wf.updated === "string" ? wf.updated : "",
  };
}

export const getIwcWorkflowDetailsOp: Operation<typeof input, IwcWorkflowDetail> = {
  name: "get_iwc_workflow_details",
  domain: "iwc",
  summary: "Get comprehensive details (inputs, outputs, full readme) for a specific IWC workflow by TRS id.",
  input,
  run,
  project: (out) => ({ message: `Retrieved details for workflow '${out.name}'` }),
};

register(getIwcWorkflowDetailsOp as AnyOperation);

export const getIwcWorkflowDetails = (i: In, ctx: GalaxyContext) => runOperation(getIwcWorkflowDetailsOp, i, ctx);
