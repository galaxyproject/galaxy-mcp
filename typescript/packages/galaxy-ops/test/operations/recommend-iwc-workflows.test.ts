import { describe, it, expect, beforeEach } from "vitest";
import { recommendIwcWorkflows, recommendIwcWorkflowsOp } from "../../src/operations/recommend-iwc-workflows";
import { __setIwcCacheForTest, __resetIwcCacheForTest, type IwcWorkflow } from "../../src/iwc-manifest";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";
import { mockClient } from "../util/mock-client";
import { runWithEnvelope } from "../../src/operations/registry";
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

/**
 * Nothing came back, and the sentence says which nothing it was.
 *
 * The other server has three answers here and this one used to have one. An empty
 * manifest is an IWC that is broken or unreachable; an intent of nothing but stop
 * words is a query to rewrite; a ranking that scored nothing is a query to make
 * more specific. All three came out as "Found 0 workflows matching your intent",
 * which is true of only the third and points an agent at the wrong fix for the
 * other two. These are the sentences the Python tool sends, in its order -- the
 * manifest is tested before the query, so an empty manifest reached with an empty
 * intent is still an empty manifest.
 */
describe("recommend_iwc_workflows says why a ranking is empty", () => {
  const envelopeFor = async (input: Record<string, unknown>) =>
    runWithEnvelope(recommendIwcWorkflowsOp as never, input as never, mockCtx);

  it("names an empty manifest rather than reporting nothing matched", async () => {
    __resetIwcCacheForTest();
    __setIwcCacheForTest([]);
    const result = await envelopeFor({ intent: "rnaseq quality control", limit: 5 });
    expect(result.message).toBe("No workflows in IWC manifest");
    expect(result.data).toEqual([]);
    expect(result.count).toBe(0);
    expect(result.pagination).toBeNull();
  });

  it("names an intent with nothing searchable in it", async () => {
    __resetIwcCacheForTest();
    __setIwcCacheForTest(fixture);
    const result = await envelopeFor({ intent: "", limit: 5 });
    expect(result.message).toBe("No searchable terms in query");
    expect(result.data).toEqual([]);
    expect(result.count).toBe(0);
    expect(result.pagination).toBeNull();
  });

  it("names an intent that is one word with an accent in it", async () => {
    __resetIwcCacheForTest();
    __setIwcCacheForTest(fixture);
    // 'café' has no searchable term in it on the other server: its tokeniser counts
    // the é as part of the word, so there is no run of ASCII letters standing alone.
    // An ASCII word boundary finds 'caf', scores it against the corpus, matches
    // nothing and reports that nothing matched -- which reads as "your query is too
    // narrow" when the truth is that the query never made it as far as the ranking.
    const result = await envelopeFor({ intent: "café", limit: 5 });
    expect(result.message).toBe("No searchable terms in query");
    expect(result.data).toEqual([]);
    expect(result.count).toBe(0);
    expect(result.pagination).toBeNull();
  });

  it("names it for stop words too, which tokenise to nothing just the same", async () => {
    __resetIwcCacheForTest();
    __setIwcCacheForTest(fixture);
    const result = await envelopeFor({ intent: "the and for with", limit: 5 });
    expect(result.message).toBe("No searchable terms in query");
  });

  it("puts the manifest first, as the other server does", async () => {
    __resetIwcCacheForTest();
    __setIwcCacheForTest([]);
    const result = await envelopeFor({ intent: "", limit: 5 });
    expect(result.message).toBe("No workflows in IWC manifest");
  });

  it("still reports a ranking that simply scored nothing the way it always did", async () => {
    __resetIwcCacheForTest();
    __setIwcCacheForTest(fixture);
    const result = await envelopeFor({ intent: "zzzz", limit: 5 });
    expect(result.message).toBe("Found 0 workflows matching your intent");
  });

  it("leaves run() answering exactly what it answered before", async () => {
    __resetIwcCacheForTest();
    __setIwcCacheForTest([]);
    // A library caller carries no collector, so nothing is recorded and nothing
    // about the value it gets back has moved.
    const out = await recommendIwcWorkflows({ intent: "rnaseq", limit: 5 }, mockCtx);
    expect(out.items).toEqual([]);
    expect(out.pagination).toMatchObject({ total: 0, returned: 0, limit: 5, hasNext: false });
  });
});

/**
 * The tokeniser runs over the manifest as well as over the intent, so a boundary rule
 * that moves moves both sides of the match at once and the ranking itself changes.
 *
 * That is not a side effect to be minimised -- it is the other server's ranking. A readme
 * that says "protéomique" offers no `prot` term there, so an intent of `prot` scores
 * nothing and comes back empty; an ASCII boundary indexes `prot` from the readme and
 * hands back a match the other server never had. The library's `run()` moves with it, as
 * it must: a ranking that disagrees with the ranking the wire reports is worse than either.
 */
describe("recommend_iwc_workflows indexes the corpus by the same rule", () => {
  const accented: IwcWorkflow[] = [
    {
      trsID: "#workflow/github.com/iwc-workflows/proteomique/main",
      definition: {
        name: "Analyse",
        annotation: "Analyse protéomique",
        tags: ["protéomique"],
        steps: {},
      },
      readme: "Une analyse protéomique complète.",
    },
    {
      trsID: "#workflow/github.com/iwc-workflows/variant-calling/main",
      definition: {
        name: "Variant Calling",
        annotation: "Call germline variants",
        tags: ["variants"],
        steps: {},
      },
      readme: "Germline short variant discovery.",
    },
    // A third document, because BM25's idf for a term in one document of two is
    // exactly zero and this tool drops a zero score: a two-entry manifest can only
    // ever rank nothing, whatever the tokeniser does.
    {
      trsID: "#workflow/github.com/iwc-workflows/assembly/main",
      definition: {
        name: "Assembly",
        annotation: "Assemble a bacterial genome",
        tags: ["assembly"],
        steps: {},
      },
      readme: "Nanopore assembly and polishing.",
    },
  ];

  beforeEach(() => {
    __resetIwcCacheForTest();
    __setIwcCacheForTest(accented);
  });

  it("no longer scores the Latin prefix of an accented word in the corpus", async () => {
    const out = await recommendIwcWorkflows({ intent: "prot", limit: 5 }, mockCtx);
    expect(out.items).toEqual([]);
  });

  it("still scores a word the accent does not touch", async () => {
    const out = await recommendIwcWorkflows({ intent: "germline variant", limit: 5 }, mockCtx);
    expect(out.items.map((w) => w.name)).toEqual(["Variant Calling"]);
  });
});
