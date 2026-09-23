import { z } from "zod";
import { fetchIwcWorkflows, enrichWorkflowResult, type EnrichedIwcWorkflow } from "../iwc-manifest";
import type { GalaxyContext } from "../context";
import { GalaxyNotFoundError } from "../errors";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

// Extend the enriched type with details-only fields
export interface IwcWorkflowDetail extends EnrichedIwcWorkflow {
  inputs: Array<{ name: string; type: string; annotation: string }>;
  outputs: Array<{ name: string; step: string }>;
  updated: string;
}

const input = {
  trsId: z.string().describe("TRS id of the IWC workflow"),
};
type In = { trsId: string };

const INPUT_TYPES = new Set(["data_input", "data_collection_input", "parameter_input"]);

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
    for (const [stepId, stepData] of Object.entries(steps)) {
      if (!stepData || typeof stepData !== "object") continue;
      const step = stepData as Record<string, unknown>;
      const stepType = typeof step["type"] === "string" ? step["type"] : "";

      if (INPUT_TYPES.has(stepType)) {
        inputs.push({
          name: typeof step["label"] === "string" ? step["label"] : `Input ${stepId}`,
          type: stepType,
          annotation: typeof step["annotation"] === "string" ? step["annotation"] : "",
        });
      }

      const workflowOutputs = step["workflow_outputs"];
      if (Array.isArray(workflowOutputs)) {
        for (const wo of workflowOutputs) {
          if (!wo || typeof wo !== "object") continue;
          const woObj = wo as Record<string, unknown>;
          const label =
            typeof woObj["label"] === "string" && woObj["label"]
              ? woObj["label"]
              : typeof woObj["output_name"] === "string"
                ? woObj["output_name"]
                : "";
          const stepLabel = typeof step["label"] === "string" ? step["label"] : `Step ${stepId}`;
          outputs.push({ name: label, step: stepLabel });
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
