import type { IwcWorkflow } from "../../src/iwc-manifest";

/**
 * A manifest entry sized like the real thing.
 *
 * The enriched summary this produces measures 1,208 B, against a live-manifest
 * median of about 1.5 KB -- on the small side, so the byte assertions over a
 * page of these are a little weaker than the real thing, not stronger. The raw
 * entry is deliberately far bigger than the summary: that gap is why the op
 * returns summaries at all.
 */
export function iwcWorkflow(n: number): IwcWorkflow {
  const slug = `rnaseq-pe-variant-${n}`;
  const steps: Record<string, unknown> = {};
  for (let s = 0; s < 24; s++) {
    steps[String(s)] = {
      id: s,
      type: s < 3 ? "data_input" : "tool",
      tool_id: `toolshed.g2.bx.psu.edu/repos/iuc/tool_${s}/tool_${s}/1.${s}.0+galaxy2`,
      tool_version: `1.${s}.0+galaxy2`,
      annotation: "Filler standing in for the parameter block a real step carries.",
      tool_state: JSON.stringify({ input: { __class__: "RuntimeValue" }, threads: 4, quality: 20 }),
    };
  }
  return {
    trsID: `#workflow/github.com/iwc-workflows/${slug}/main`,
    definition: {
      name: `Paired-end RNA-seq differential expression (${slug})`,
      annotation:
        "Quality control, trimming, HISAT2 alignment against a reference genome, featureCounts " +
        "quantification and DESeq2 differential expression for paired-end Illumina RNA-seq data.",
      tags: ["transcriptomics", "rnaseq", "differential-expression", "illumina", "paired-end"],
      license: "MIT",
      creator: [
        { name: "Workflow Author", identifier: "https://orcid.org/0000-0002-1825-0097" },
        { name: "Second Author", identifier: "https://orcid.org/0000-0001-5109-3700" },
      ],
      steps,
    },
    readme:
      "# RNA-seq PE\n\nThis workflow takes a collection of paired-end reads, runs FastQC and " +
      "Cutadapt for quality control, aligns with HISAT2, counts features with featureCounts and " +
      "calls differential expression with DESeq2. It expects a reference genome to be available " +
      "on the target Galaxy server and a sample sheet describing the experimental factors.",
    categories: ["Transcriptomics"],
    updated: "2026-08-04",
  };
}

export const iwcManifest = (count: number): IwcWorkflow[] =>
  Array.from({ length: count }, (_, k) => iwcWorkflow(k + 1));

/**
 * A manifest with two distinct topics, alternating.
 *
 * A corpus where every entry says the same thing gives BM25 an inverse
 * document frequency of zero for every term, so nothing scores above zero and
 * a relevance test silently measures nothing. Two topics keep the ranking real.
 */
export function iwcMixedManifest(count: number): IwcWorkflow[] {
  return Array.from({ length: count }, (_, k) => {
    const wf = iwcWorkflow(k + 1);
    if (k % 2 === 0) return wf;
    const slug = `chipseq-peaks-${k + 1}`;
    return {
      ...wf,
      trsID: `#workflow/github.com/iwc-workflows/${slug}/main`,
      definition: {
        ...wf.definition,
        name: `ChIP-seq peak calling (${slug})`,
        annotation:
          "Quality control, Bowtie2 alignment, MACS2 peak calling and annotation for ChIP-seq " +
          "experiments with matched input controls.",
        tags: ["epigenomics", "chipseq", "peak-calling", "illumina"],
      },
      readme:
        "# ChIP-seq\n\nAligns ChIP and input libraries with Bowtie2, calls peaks with MACS2 and " +
        "annotates them against a reference gene set. Expects matched input controls.",
      categories: ["Epigenetics"],
    };
  });
}

/**
 * A manifest entry whose summary is as big as a heavily documented workflow's.
 *
 * The summary is what the listings return, so this pads the fields that survive
 * into it rather than the definition, which does not.
 */
export function hugeIwcWorkflow(n: number): IwcWorkflow {
  const base = iwcWorkflow(n);
  const CJK = "\u30b2\u30ce\u30e0\u89e3\u6790\u30d1\u30a4\u30d7\u30e9\u30a4\u30f3".repeat(3);
  return {
    ...base,
    definition: {
      ...base.definition,
      name: `${CJK} ${n}`,
      annotation: "\u3053\u306e\u30ef\u30fc\u30af\u30d5\u30ed\u30fc\u306e\u8aac\u660e\u3002".repeat(60),
    },
    readme: base.readme + "\n\n" + "\u8a73\u7d30\u306a\u624b\u9806\u3002".repeat(120),
  };
}

export const hugeIwcManifest = (count: number): IwcWorkflow[] =>
  Array.from({ length: count }, (_, k) => hugeIwcWorkflow(k + 1));
