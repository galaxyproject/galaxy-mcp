import { z } from "zod";
import { fetchIwcWorkflows, enrichWorkflowResult, type EnrichedIwcWorkflow } from "../iwc-manifest";
import type { GalaxyContext } from "../context";
import { GalaxyNotFoundError } from "../errors";
import { pyGet } from "../python-values";
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

async function run(i: In, _ctx: GalaxyContext): Promise<IwcWorkflowDetail> {
  const workflows = await fetchIwcWorkflows();
  const wf = workflows.find((w) => w.trsID === i.trsId);
  // server.py, get_iwc_workflow_details: this refusal is raised INSIDE its try, so the
  // tool's own sentence wraps it -- the same prefix a refused manifest fetch gets.
  if (!wf) {
    throw new GalaxyNotFoundError(
      "Failed to get IWC workflow details: " +
        `Workflow with trsID '${i.trsId}' not found in IWC manifest. ` +
        "Check the trsID format and use search_iwc_workflows() to find valid IDs.",
    );
  }

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
          name: pyGet(step, "label", `Input ${stepId}`) as string | null,
          type: stepType,
          annotation: pyGet(step, "annotation", "") as string | null,
        });
      }

      const workflowOutputs = step["workflow_outputs"];
      if (Array.isArray(workflowOutputs)) {
        for (const wo of workflowOutputs) {
          if (!wo || typeof wo !== "object") continue;
          const woObj = wo as Record<string, unknown>;
          const label = pyGet(woObj, "label", pyGet(woObj, "output_name", ""));
          const stepLabel = pyGet(step, "label", `Step ${stepId}`);
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
  // server.py, get_iwc_workflow_details: its refusal for a trsID nobody has is raised INSIDE
  // its try, so the tool's own sentence wraps it too -- see the throw site.
  failure: {
    shape: "raise-for-status",
    sentence: (text) => `Failed to get IWC workflow details: ${text}`,
  },
};

register(getIwcWorkflowDetailsOp as AnyOperation);

export const getIwcWorkflowDetails = (i: In, ctx: GalaxyContext) => runOperation(getIwcWorkflowDetailsOp, i, ctx);
