/**
 * Galaxy index entries sized like the ones a public server returns, and the
 * oversized ones the output budget has to cut.
 *
 * A usegalaxy.org /api/tools entry is fat -- a toolshed id, EDAM terms, a
 * config path and a whole tool_shed_repository block. Measured: tool index entry
 * 951 B, user tool entry 872 B, workflow index entry 497 B. All three sit a
 * little under the real thing, so a byte assertion over them is conservative.
 *
 * The `huge*` builders are what the budget tests use. A ceiling is a limit on
 * the request, not a promise about the answer, so the only way to show the
 * promise is kept is to ask for the ceiling against records no count could have
 * allowed for. The names are multi-byte on purpose: a CJK name is three bytes a
 * character, and a budget measured in characters would be half the truth.
 */
const CJK = "\u30b2\u30ce\u30e0\u89e3\u6790\u30d1\u30a4\u30d7\u30e9\u30a4\u30f3".repeat(3);

export function toolIndexEntry(n: number): Record<string, unknown> {
  const owner = "iuc";
  const repo = `bwa_mem_variant_${n}`;
  return {
    id: `toolshed.g2.bx.psu.edu/repos/${owner}/${repo}/${repo}/0.7.17.${n}+galaxy1`,
    name: `Map with BWA-MEM variant ${n}`,
    version: `0.7.17.${n}+galaxy1`,
    description: "map medium and long reads (> 100 bp) against a reference genome",
    labels: [],
    edam_operations: ["operation_0523", "operation_3198"],
    edam_topics: ["topic_0102", "topic_3168"],
    hidden: "",
    is_workflow_compatible: true,
    xrefs: [{ value: "bio.tools/bwa", reftype: "bio.tools" }],
    config_file: `/srv/galaxy/shed_tools/toolshed.g2.bx.psu.edu/repos/${owner}/${repo}/abc123def456/${repo}/bwa-mem.xml`,
    link: `/tool_runner?tool_id=toolshed.g2.bx.psu.edu%2Frepos%2F${owner}%2F${repo}`,
    min_width: -1,
    target: "galaxy_main",
    panel_section_id: "ngs_mapping",
    panel_section_name: "Genomics Analysis / Mapping",
    form_style: "regular",
    tool_shed_repository: {
      name: repo,
      owner,
      changeset_revision: "abc123def456",
      tool_shed: "toolshed.g2.bx.psu.edu",
    },
  };
}

export const toolIndex = (count: number) => Array.from({ length: count }, (_, k) => toolIndexEntry(k + 1));

/** A /api/unprivileged_tools entry, which embeds the tool's own representation. */
export function userToolEntry(n: number): Record<string, unknown> {
  return {
    id: `ut${n}`.padStart(16, "0"),
    uuid: `00000000-0000-4000-8000-${String(n).padStart(12, "0")}`,
    tool_id: `user_tool_${n}`,
    active: true,
    create_time: "2026-07-01T10:00:00.000000",
    update_time: "2026-09-01T10:00:00.000000",
    representation: {
      id: `user_tool_${n}`,
      name: `Summarise counts table ${n}`,
      version: "0.1.0",
      description: "Collapse a counts matrix to per-condition means and write a tidy table.",
      container: "quay.io/biocontainers/pandas:2.2.1",
      shell_command: "python3 /scripts/summarise.py --input '$input' --output '$output'",
      inputs: [
        { name: "input", type: "data", format: ["tabular"], label: "Counts matrix" },
        { name: "factor", type: "text", label: "Grouping column", value: "condition" },
      ],
      outputs: [{ name: "output", type: "data", format: "tabular", label: "Per-condition means" }],
      citations: [],
      help: "Reads a counts matrix, groups the columns by the chosen factor and emits the means.",
    },
  };
}

export const userToolIndex = (count: number) => Array.from({ length: count }, (_, k) => userToolEntry(k + 1));

/** A StoredWorkflow index entry from /api/workflows. */
export function workflowIndexEntry(n: number): Record<string, unknown> {
  // Padded with zeros, not with a hex digit: "c" as the pad makes 0x3 and 0xc3
  // the same sixteen characters, and a page-walk test cannot see a duplicate it
  // was handed by its own fixture.
  return {
    id: n.toString(16).padStart(16, "0"),
    name: `Paired-end RNA-seq differential expression ${n}`,
    owner: "researcher",
    create_time: "2026-05-12T14:03:22.000000",
    update_time: "2026-09-02T08:41:10.000000",
    published: false,
    importable: false,
    deleted: false,
    hidden: false,
    tags: ["rnaseq", "deseq2"],
    latest_workflow_uuid: `00000000-0000-4000-8000-${String(n).padStart(12, "0")}`,
    annotations: ["Standard lab pipeline for paired-end libraries."],
    url: `/api/workflows/${n.toString(16).padStart(16, "0")}`,
    number_of_steps: 24,
    show_in_tool_panel: false,
  };
}

export const workflowIndex = (count: number) => Array.from({ length: count }, (_, k) => workflowIndexEntry(k + 1));

/** A tool panel: sections holding slim leaf nodes, with a divider label thrown in. */
export function toolPanel(sections: number, toolsPerSection: number): Array<Record<string, unknown>> {
  return Array.from({ length: sections }, (_, s) => ({
    id: `section_${s + 1}`,
    name: `Genomics Analysis / Section ${s + 1}`,
    model_class: "ToolSection",
    elems: [
      { id: `section_${s + 1}_label`, name: "Divider", model_class: "ToolSectionLabel" },
      ...Array.from({ length: toolsPerSection }, (_, t) => ({
        id: `toolshed.g2.bx.psu.edu/repos/iuc/tool_${s}_${t}/tool_${s}_${t}/1.2.3+galaxy0`,
        name: `Section ${s + 1} tool ${t + 1}`,
        description: "map medium and long reads against a reference genome and report alignments",
        versions: ["1.2.3+galaxy0", "1.2.2+galaxy1"],
        model_class: "Tool",
      })),
    ],
  }));
}

/** A tool index entry no cap could have sized for. */
export function hugeToolIndexEntry(n: number): Record<string, unknown> {
  return {
    ...toolIndexEntry(n),
    name: `${CJK} ${n}`,
    description: "\u3053\u306e\u30c4\u30fc\u30eb\u306e\u8aac\u660e\u3002".repeat(40),
  };
}

export const hugeToolIndex = (count: number) =>
  Array.from({ length: count }, (_, k) => hugeToolIndexEntry(k + 1));

/** A user tool whose embedded representation is as big as they get. */
export function hugeUserToolEntry(n: number): Record<string, unknown> {
  const entry = userToolEntry(n) as Record<string, unknown>;
  const representation = { ...(entry.representation as Record<string, unknown>) };
  representation.shell_command = "multiqc --no-data-dir " + "-x arg ".repeat(340);
  representation.help = "\u8a73\u7d30\u306a\u8aac\u660e\u3002".repeat(80);
  return { ...entry, name: `${CJK} ${n}`, representation };
}

export const hugeUserToolIndex = (count: number) =>
  Array.from({ length: count }, (_, k) => hugeUserToolEntry(k + 1));

/** A workflow index entry carrying the annotation a heavily documented workflow has. */
export function hugeWorkflowEntry(n: number): Record<string, unknown> {
  return {
    ...workflowIndexEntry(n),
    name: `${CJK} ${n}`,
    annotations: ["\u6ce8\u91c8\u3002".repeat(100)],
  };
}

export const hugeWorkflowIndex = (count: number) =>
  Array.from({ length: count }, (_, k) => hugeWorkflowEntry(k + 1));

/** A panel whose sections and tools are both oversized. */
export function hugeToolPanel(sections: number, toolsPerSection: number): Array<Record<string, unknown>> {
  return Array.from({ length: sections }, (_, s) => ({
    id: `section_${s + 1}`,
    name: `${CJK} ${s + 1}`,
    model_class: "ToolSection",
    elems: Array.from({ length: toolsPerSection }, (_, k) => ({
      ...hugeToolIndexEntry(s * toolsPerSection + k + 1),
      model_class: "Tool",
      versions: ["1.2.3+galaxy0", "1.2.2+galaxy1"],
    })),
  }));
}
