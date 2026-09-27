import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  fetchIwcWorkflows,
  enrichWorkflowResult,
  cleanReadmeSummary,
  extractToolNamesFromSteps,
  __resetIwcCacheForTest,
  __setIwcCacheForTest,
  type IwcWorkflow,
} from "../src/iwc-manifest";

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

const FIXTURE_WF_1: IwcWorkflow = {
  trsID: "#workflow/github.com/iwc-workflows/rnaseq-pe/main",
  definition: {
    name: "RNA-Seq PE",
    annotation: "Paired-end RNA-seq analysis",
    tags: ["transcriptomics", "rnaseq"],
    steps: {
      "0": { type: "data_collection_input", label: "PE reads", annotation: "Paired-end reads", tool_id: "" },
      "1": { tool_id: "toolshed.g2.bx.psu.edu/repos/iuc/fastqc/fastqc/0.73", label: "FastQC" },
      "2": { tool_id: "toolshed.g2.bx.psu.edu/repos/iuc/hisat2/hisat2/2.2.1", label: "HISAT2" },
      "3": {
        tool_id: "toolshed.g2.bx.psu.edu/repos/iuc/hisat2/hisat2/2.2.1",
        label: "HISAT2 again",
        workflow_outputs: [{ label: "alignments", output_name: "output" }],
      },
    },
    creator: [{ name: "IWC Team", identifier: "https://orcid.org/0000-0001-0000-0000" }],
    license: "MIT",
  },
  readme:
    "# RNA-Seq Paired-End\n\nThis workflow performs paired-end RNA-seq analysis.\nIt uses FastQC for QC and HISAT2 for alignment.",
  categories: ["Transcriptomics"],
  updated: "2024-01-15",
};

const FIXTURE_WF_2: IwcWorkflow = {
  trsID: "#workflow/github.com/iwc-workflows/assembly-ont/main",
  definition: {
    name: "ONT Assembly",
    annotation: "Oxford Nanopore assembly",
    tags: ["assembly", "nanopore"],
    steps: {
      "0": { type: "data_input", label: "ONT reads", annotation: "" },
      "1": { tool_id: "toolshed.g2.bx.psu.edu/repos/iuc/flye/flye/2.9.1", label: "Flye" },
    },
    creator: [],
    license: "Apache-2.0",
  },
  readme: "## ONT Assembly\n\nAssembles genomes from Oxford Nanopore reads.",
  categories: ["Assembly"],
  updated: "2024-03-10",
};

const MANIFEST_ENTRIES = [
  { workflows: [FIXTURE_WF_1] },
  { workflows: [FIXTURE_WF_2] },
];

beforeEach(() => {
  __resetIwcCacheForTest();
  vi.restoreAllMocks();
});

// ---------------------------------------------------------------------------
// cleanReadmeSummary
// ---------------------------------------------------------------------------

describe("cleanReadmeSummary", () => {
  it("strips markdown headers and trims leading blank lines", () => {
    const s = cleanReadmeSummary("# Title\n\nSome text here.");
    expect(s).toBe("Some text here.");
    expect(s).not.toContain("#");
  });

  it("collapses interior whitespace", () => {
    const s = cleanReadmeSummary("line one\n  line two\n");
    expect(s).toBe("line one line two");
  });

  it("truncates to ~300 chars at a word boundary with ellipsis", () => {
    const long = "word ".repeat(100); // 500 chars
    const s = cleanReadmeSummary(long);
    expect(s.length).toBeLessThanOrEqual(300);
    expect(s.endsWith("...")).toBe(true);
    // should not cut mid-word
    const before = s.slice(0, s.length - 3);
    expect(before.trimEnd()).toBe(before.trimEnd().replace(/\s$/, ""));
  });

  it("returns empty string for empty or undefined input", () => {
    expect(cleanReadmeSummary("")).toBe("");
    expect(cleanReadmeSummary(undefined)).toBe("");
  });
});

/**
 * The same readmes, summarised by the other server.
 *
 * `readme_summary` is a field of `data` for three of the budgeted IWC listings, so a
 * summary that differs is an envelope that differs -- and it is decided entirely by
 * three JavaScript built-ins whose Python namesakes draw their lines elsewhere:
 * `trim` and `split(/\s+/)` disagree with `strip()` and `split()` about U+FEFF and the
 * C0 information separators, and `length` counts UTF-16 code units where `len()` counts
 * code points.
 *
 * Each expectation below came off the installed interpreter, from this script:
 *
 *   from galaxy_mcp.workflow_inputs import _clean_readme_summary as c
 *   c("﻿# Title\n\nBody text here.")   ->  '﻿# Title Body text here.'
 *   c("# Title\nBody text here.")     ->  'Body text here.'
 *   c("# Title\nBody text here.")     ->  'Body text here.'
 *   c("alpha﻿beta gamma")              ->  'alpha﻿beta gamma'
 *   c("alpha beta gamma")              ->  'alpha beta gamma'
 *   c(("word " * 55) + "\U0001F9EC gene " + ("tail " * 20))
 *                                           ->  299 code points, ending 'tail tail tail...'
 */
describe("cleanReadmeSummary reads whitespace and length the way the other server does", () => {
  it("keeps a heading that a byte order mark pushed off the margin", () => {
    // `strip()` does not remove U+FEFF, so the line never looks like a heading there and
    // the summary opens with it. `trimStart` does remove it, and dropped the line.
    expect(cleanReadmeSummary("﻿# Title\n\nBody text here.")).toBe(
      "﻿# Title Body text here.",
    );
  });

  it("drops a heading behind a next-line character", () => {
    expect(cleanReadmeSummary("# Title\nBody text here.")).toBe("Body text here.");
  });

  it("drops a heading behind a unit separator", () => {
    expect(cleanReadmeSummary("# Title\nBody text here.")).toBe("Body text here.");
  });

  it("leaves a byte order mark inside a word alone rather than collapsing it", () => {
    expect(cleanReadmeSummary("alpha﻿beta gamma")).toBe("alpha﻿beta gamma");
  });

  it("still collapses a no-break space, which both call whitespace", () => {
    expect(cleanReadmeSummary("alpha beta gamma")).toBe("alpha beta gamma");
  });

  it("measures and cuts the summary in code points, not UTF-16 units", () => {
    const readme = "word ".repeat(55) + "\u{1F9EC} gene " + "tail ".repeat(20);
    const summary = cleanReadmeSummary(readme);
    expect(summary).toBe("word ".repeat(55) + "\u{1F9EC} gene tail tail tail...");
    expect(Array.from(summary)).toHaveLength(299);
    // The emoji survives whole: a cut by code unit can leave half of one behind.
    expect(summary).toContain("\u{1F9EC}");
  });
});

// ---------------------------------------------------------------------------
// extractToolNamesFromSteps
// ---------------------------------------------------------------------------

describe("extractToolNamesFromSteps", () => {
  it("extracts deduplicated tool names from toolshed IDs", () => {
    const steps = {
      "0": { tool_id: "toolshed.g2.bx.psu.edu/repos/iuc/fastqc/fastqc/0.73" },
      "1": { tool_id: "toolshed.g2.bx.psu.edu/repos/iuc/hisat2/hisat2/2.2.1" },
      "2": { tool_id: "toolshed.g2.bx.psu.edu/repos/iuc/hisat2/hisat2/2.2.1" }, // duplicate
    };
    const tools = extractToolNamesFromSteps(steps);
    expect(tools).toEqual(["fastqc", "hisat2"]);
  });

  it("uses whole id when no slash present", () => {
    const steps = { "0": { tool_id: "simple_tool" } };
    expect(extractToolNamesFromSteps(steps)).toEqual(["simple_tool"]);
  });

  it("skips steps without tool_id", () => {
    const steps = { "0": { type: "data_input" } };
    expect(extractToolNamesFromSteps(steps)).toEqual([]);
  });

  it("handles array steps (returns empty -- same as Python)", () => {
    // Python: steps.values() on a list gives list elements but _extract only
    // called when isinstance(steps, dict) -- returns [] for arrays
    const result = extractToolNamesFromSteps([]);
    expect(result).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// enrichWorkflowResult
// ---------------------------------------------------------------------------

describe("enrichWorkflowResult", () => {
  it("maps fields correctly from fixture wf1", () => {
    const e = enrichWorkflowResult(FIXTURE_WF_1);
    expect(e.trsID).toBe(FIXTURE_WF_1.trsID);
    expect(e.name).toBe("RNA-Seq PE");
    expect(e.description).toBe("Paired-end RNA-seq analysis");
    expect(e.tags).toEqual(["transcriptomics", "rnaseq"]);
    expect(e.categories).toEqual(["Transcriptomics"]);
    expect(e.license).toBe("MIT");
    expect(e.step_count).toBe(4); // 4 keys in steps dict
    expect(e.tools_used).toEqual(["fastqc", "hisat2"]); // deduped
  });

  it("maps author name + orcid (identifier field)", () => {
    const e = enrichWorkflowResult(FIXTURE_WF_1);
    expect(e.authors).toEqual([{ name: "IWC Team", orcid: "https://orcid.org/0000-0001-0000-0000" }]);
  });

  it("authors is empty array when creator is empty list", () => {
    const e = enrichWorkflowResult(FIXTURE_WF_2);
    expect(e.authors).toEqual([]);
  });

  it("readme_summary is a cleaned short excerpt (no #)", () => {
    const e = enrichWorkflowResult(FIXTURE_WF_1);
    expect(e.readme_summary).not.toContain("#");
    expect(e.readme_summary.length).toBeGreaterThan(0);
    expect(e.readme_summary).toContain("RNA-seq analysis");
  });

  it("does not include readme by default", () => {
    const e = enrichWorkflowResult(FIXTURE_WF_1);
    expect("readme" in e).toBe(false);
  });

  it("includes full readme when fullReadme=true", () => {
    const e = enrichWorkflowResult(FIXTURE_WF_1, { fullReadme: true });
    expect(e.readme).toBe(FIXTURE_WF_1.readme);
  });
});

// ---------------------------------------------------------------------------
// fetchIwcWorkflows -- memoization
// ---------------------------------------------------------------------------

describe("fetchIwcWorkflows", () => {
  it("flattens manifest entries from fetch and memoizes (fetch called only once)", async () => {
    let callCount = 0;
    vi.stubGlobal("fetch", async (url: string) => {
      expect(url).toBe("https://iwc.galaxyproject.org/workflow_manifest.json");
      callCount++;
      return {
        ok: true,
        json: async () => MANIFEST_ENTRIES,
      };
    });

    const first = await fetchIwcWorkflows();
    const second = await fetchIwcWorkflows();

    expect(first).toHaveLength(2);
    expect(second).toHaveLength(2);
    expect(callCount).toBe(1); // memoized
  });

  it("__setIwcCacheForTest primes the cache without network", async () => {
    __setIwcCacheForTest([FIXTURE_WF_1]);
    const wfs = await fetchIwcWorkflows();
    expect(wfs).toHaveLength(1);
    expect(wfs[0].trsID).toBe(FIXTURE_WF_1.trsID);
  });

  it("__resetIwcCacheForTest clears so next fetch re-fetches", async () => {
    __setIwcCacheForTest([FIXTURE_WF_1]);
    __resetIwcCacheForTest();

    let called = false;
    vi.stubGlobal("fetch", async () => {
      called = true;
      return { ok: true, json: async () => MANIFEST_ENTRIES };
    });

    await fetchIwcWorkflows();
    expect(called).toBe(true);
  });
});
