import { describe, it, expect, beforeEach } from "vitest";
import { recommendIwcWorkflows, recommendIwcWorkflowsOp } from "../../src/operations/recommend-iwc-workflows";
import { __setIwcCacheForTest, __resetIwcCacheForTest, type IwcWorkflow } from "../../src/iwc-manifest";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";
import { mockClient } from "../util/mock-client";
import { iwcMixedManifest } from "../util/iwc-fixture";

const ctxWith = (client: ReturnType<typeof mockClient>): GalaxyContext => ({ client, poll: DEFAULT_POLL });
const mockCtx = ctxWith(mockClient({}));

const fixture: IwcWorkflow[] = [
  {
    trsID: "#workflow/github.com/iwc-workflows/rna-seq-alignment/main",
    definition: {
      name: "RNA-seq Alignment",
      annotation: "Align RNA-seq reads to a reference genome using STAR",
      tags: ["rna-seq", "alignment"],
      steps: {
        "0": { type: "tool", tool_id: "toolshed.g2.bx.psu.edu/repos/iuc/star/rna_star/2.7" },
      },
    },
    readme: "Aligns RNA-seq data using STAR aligner and produces BAM files.",
  },
  {
    trsID: "#workflow/github.com/iwc-workflows/maxquant/main",
    definition: {
      name: "MaxQuant Proteomics",
      annotation: "Quantitative proteomics analysis using MaxQuant",
      tags: ["proteomics", "mass-spectrometry"],
      steps: {
        "0": { type: "tool", tool_id: "toolshed.g2.bx.psu.edu/repos/iuc/maxquant/maxquant/1.6" },
      },
    },
    readme: "Run MaxQuant for label-free proteomics quantification.",
  },
  {
    trsID: "#workflow/github.com/iwc-workflows/gatk-variant-calling/main",
    definition: {
      name: "GATK Variant Calling",
      annotation: "Call germline variants using GATK HaplotypeCaller",
      tags: ["variant-calling", "gatk"],
      steps: {
        "0": { type: "tool", tool_id: "toolshed.g2.bx.psu.edu/repos/iuc/gatk4/gatk4_haplotypecaller/4.2" },
      },
    },
    readme: "Germline short variant discovery using GATK best practices.",
  },
];

describe("recommendIwcWorkflows", () => {
  beforeEach(() => {
    __resetIwcCacheForTest();
    __setIwcCacheForTest(fixture);
  });

  it("ranks the matching workflow first when intent matches its name", async () => {
    const { items: results } = await recommendIwcWorkflows({ intent: "rna seq alignment star" }, mockCtx);
    expect(results.length).toBeGreaterThan(0);
    expect(results[0].name).toBe("RNA-seq Alignment");
  });

  it("excludes workflows with a zero score (no query term matches)", async () => {
    const { items: results } = await recommendIwcWorkflows({ intent: "maxquant proteomics quantification" }, mockCtx);
    const names = results.map((r) => r.name);
    expect(names).toContain("MaxQuant Proteomics");
    expect(names[0]).toBe("MaxQuant Proteomics");
  });

  it("returns empty array for empty intent", async () => {
    const out = await recommendIwcWorkflows({ intent: "" }, mockCtx);
    expect(out.items).toEqual([]);
    expect(out.pagination).toMatchObject({ total: 0, returned: 0, hasNext: false });
  });

  it("returns empty array for stopword-only intent", async () => {
    const out = await recommendIwcWorkflows({ intent: "the and for with" }, mockCtx);
    expect(out.items).toEqual([]);
    expect(out.pagination).toMatchObject({ total: 0, returned: 0 });
  });

  it("respects the limit parameter", async () => {
    const { items: results } = await recommendIwcWorkflows({ intent: "alignment calling analysis", limit: 2 }, mockCtx);
    expect(results.length).toBeLessThanOrEqual(2);
  });

  it("each result has a numeric match_score", async () => {
    const { items: results } = await recommendIwcWorkflows({ intent: "rna seq reads alignment" }, mockCtx);
    expect(results.length).toBeGreaterThan(0);
    for (const r of results) {
      expect(typeof r.match_score).toBe("number");
      expect(r.match_score).toBeGreaterThan(0);
    }
  });
});

describe("recommend_iwc_workflows pagination", () => {
  beforeEach(() => {
    __resetIwcCacheForTest();
    __setIwcCacheForTest(fixture);
  });

  it("says how many workflows actually scored without offering an unusable cursor", async () => {
    const out = await recommendIwcWorkflows({ intent: "alignment calling analysis", limit: 1 }, mockCtx);
    expect(out.items).toHaveLength(1);
    // The agent can tell "1 of 1" from "1 of 40", but ranking is top-N,
    // not a window it can continue by passing an offset.
    expect(out.pagination.total).toBeGreaterThan(1);
    expect(out.pagination).toMatchObject({ returned: 1, limit: 1, hasNext: false });
    expect(out.pagination).not.toHaveProperty("offset");
    expect(out.pagination).not.toHaveProperty("nextOffset");
    expect(out.pagination.helperText).toContain("Refine the intent");
  });

  it("calls itself the last page when every match fits", async () => {
    const out = await recommendIwcWorkflows({ intent: "maxquant proteomics quantification" }, mockCtx);
    expect(out.pagination).toMatchObject({ total: out.items.length, hasNext: false });
  });

  it("declares its default of 5 on the advertised schema", () => {
    expect(recommendIwcWorkflowsOp.input.limit.parse(undefined)).toBe(5);
  });

  it("does not advertise an offset input", () => {
    expect(recommendIwcWorkflowsOp.input).not.toHaveProperty("offset");
  });

  it("rejects a limit over the cap, naming the ceiling", async () => {
    await expect(recommendIwcWorkflows({ intent: "rna-seq", limit: 26 }, {} as any)).rejects.toThrow(/at most 25/);
  });

  it("rejects a limit below one rather than silently dropping a result", async () => {
    await expect(recommendIwcWorkflows({ intent: "rna-seq", limit: 0 }, {} as any)).rejects.toThrow(/at least 1/);
  });
});
