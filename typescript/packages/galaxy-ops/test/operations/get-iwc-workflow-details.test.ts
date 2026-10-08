import { describe, it, expect, beforeEach } from "vitest";
import { getIwcWorkflowDetailsOp, getIwcWorkflowDetails } from "../../src/operations/get-iwc-workflow-details";
import { __resetIwcCacheForTest, __setIwcCacheForTest, type IwcWorkflow } from "../../src/iwc-manifest";
import { GalaxyNotFoundError } from "../../src/errors";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";
import { mockClient } from "../util/mock-client";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

const FULL_WF: IwcWorkflow = {
  trsID: "#workflow/github.com/iwc-workflows/rnaseq-pe/main",
  definition: {
    name: "RNA-Seq PE",
    annotation: "Paired-end RNA-seq analysis",
    tags: ["transcriptomics"],
    steps: {
      "0": {
        type: "data_collection_input",
        label: "PE reads",
        annotation: "Paired reads input",
      },
      "1": {
        type: "parameter_input",
        label: "Genome build",
        annotation: "Reference genome",
      },
      "2": {
        tool_id: "toolshed.g2.bx.psu.edu/repos/iuc/fastqc/fastqc/0.73",
        label: "FastQC",
        workflow_outputs: [{ label: "QC report", output_name: "html_file" }],
      },
      "3": {
        tool_id: "toolshed.g2.bx.psu.edu/repos/iuc/hisat2/hisat2/2.2.1",
        label: "HISAT2",
        workflow_outputs: [
          { label: "", output_name: "alignments" },
          { label: "splice sites", output_name: "splice_file" },
        ],
      },
    },
    creator: [{ name: "IWC", identifier: "https://orcid.org/0000-0000-0000-0001" }],
    license: "MIT",
  },
  readme: "# RNA-Seq\n\nPaired-end RNA-seq workflow documentation.",
  categories: ["Transcriptomics"],
  updated: "2024-01-15",
};

beforeEach(() => __resetIwcCacheForTest());

describe("get_iwc_workflow_details", () => {
  it("returns enriched result with inputs and outputs extracted", async () => {
    __setIwcCacheForTest([FULL_WF]);
    const ctx = ctxWith(mockClient({}));
    const out = await getIwcWorkflowDetails({ trsId: FULL_WF.trsID }, ctx);

    expect(out.trsID).toBe(FULL_WF.trsID);
    expect(out.name).toBe("RNA-Seq PE");
    expect(out.license).toBe("MIT");
    expect(out.authors).toEqual([{ name: "IWC", orcid: "https://orcid.org/0000-0000-0000-0001" }]);
    expect(out.updated).toBe("2024-01-15");

    // full readme included
    expect(out.readme).toBe(FULL_WF.readme);

    // inputs: steps with type in input types
    expect(out.inputs).toHaveLength(2);
    const inputNames = out.inputs.map((i) => i.name);
    expect(inputNames).toContain("PE reads");
    expect(inputNames).toContain("Genome build");
    expect(out.inputs[0].type).toMatch(/input/);

    // outputs: workflow_outputs from tool steps
    expect(out.outputs.length).toBeGreaterThanOrEqual(3);
    const outputNames = out.outputs.map((o) => o.name);
    expect(outputNames).toContain("QC report");
    expect(outputNames).toContain("splice sites");
    // an empty label is the value, as it is on the other server; only an absent
    // one falls back to output_name
    expect(outputNames).toContain("");
    expect(outputNames).not.toContain("alignments");
  });

  it("passes a null or empty label through and defaults only an absent one", async () => {
    __setIwcCacheForTest([
      {
        ...FULL_WF,
        trsID: "#workflow/github.com/iwc-workflows/nulls/main",
        definition: {
          ...FULL_WF.definition,
          steps: {
            "1": {
              type: "data_input",
              label: null,
              annotation: null,
              workflow_outputs: [
                { label: null, output_name: "output" },
                { label: "", output_name: "out2" },
                { output_name: "out3" },
                {},
              ],
            },
            "2": { type: "tool", tool_id: "t", workflow_outputs: [{ label: "named" }] },
            "3": { type: "parameter_input" },
          },
        },
      },
    ]);
    const out = await getIwcWorkflowDetails(
      { trsId: "#workflow/github.com/iwc-workflows/nulls/main" },
      ctxWith(mockClient({})),
    );
    expect(out.inputs).toEqual([
      { name: null, type: "data_input", annotation: null },
      { name: "Input 3", type: "parameter_input", annotation: "" },
    ]);
    expect(out.outputs).toEqual([
      { name: null, step: null },
      { name: "", step: null },
      { name: "out3", step: null },
      { name: "", step: null },
      { name: "named", step: "Step 2" },
    ]);
  });

  it("throws GalaxyNotFoundError when trsId is not in manifest", async () => {
    __setIwcCacheForTest([FULL_WF]);
    const ctx = ctxWith(mockClient({}));
    await expect(getIwcWorkflowDetails({ trsId: "nonexistent" }, ctx)).rejects.toBeInstanceOf(GalaxyNotFoundError);
  });

  it("project message names the workflow", async () => {
    __setIwcCacheForTest([FULL_WF]);
    const out = await getIwcWorkflowDetails({ trsId: FULL_WF.trsID }, ctxWith(mockClient({})));
    const meta = getIwcWorkflowDetailsOp.project!(out, { trsId: FULL_WF.trsID });
    expect(meta.message).toContain("RNA-Seq PE");
  });
});
