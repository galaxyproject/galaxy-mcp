/**
 * Made-up results for looking at the cards without a Galaxy: enough histories and items to page
 * through, one item in each state a card treats differently, and names a card must show as text.
 *
 * `answer` stands in for the server. It returns the envelope a read would, cut to the page that
 * was asked for, so "Show more" on a previewed card fetches a real second page.
 */

const DAY = 24 * 60 * 60 * 1000;
const day = (n: number) => new Date(Date.UTC(2026, 0, 31) - n * DAY).toISOString().replace("Z", "");

export const histories = [
  "RNA-seq: liver vs. kidney",
  "16S amplicon, soil cores",
  "Variant calling, trio",
  "ChIP-seq H3K27ac",
  "",
  "Assembly, long reads",
  "Proteomics, label-free",
  "Single-cell, PBMC 3k",
  "Metagenomics, wastewater",
  "ATAC-seq time course",
  "<b>Named like markup</b>",
  "A history with a name long enough that it has to be cut short where the card runs out of room",
].map((name, index) => ({ id: `h${index + 1}`, name, count: [24, 61, 18, 33, 0, 7][index % 6]!, update_time: day(index * 3) }));

const dataset = (hid: number, name: string, extension: string, state: string, extra: Record<string, unknown> = {}) => ({
  id: `d${hid}`,
  hid,
  name,
  history_content_type: "dataset",
  extension,
  state,
  deleted: false,
  visible: true,
  file_size: hid * 91_337,
  create_time: day(30 - hid),
  update_time: day(30 - hid),
  ...extra,
});

const collection = (hid: number, name: string, collection_type: string, element_count: number, populated_state = "ok") => ({
  id: `c${hid}`,
  hid,
  name,
  history_content_type: "dataset_collection",
  collection_type,
  element_count,
  populated_state,
  deleted: false,
  visible: true,
});

/** One history's items, oldest first. */
export const contents = [
  dataset(1, "reads_R1.fastq.gz", "fastqsanger.gz", "ok"),
  dataset(2, "reads_R2.fastq.gz", "fastqsanger.gz", "ok"),
  collection(3, "Paired reads", "list:paired", 6),
  dataset(4, "reference.fasta", "fasta", "ok"),
  dataset(5, "genes.gtf", "gtf", "ok"),
  dataset(6, "Trimmed reads R1", "fastqsanger.gz", "ok"),
  dataset(7, "Trimmed reads R2", "fastqsanger.gz", "error"),
  dataset(8, "FastQC on R1", "html", "ok"),
  dataset(9, "An upload still arriving", "auto", "upload"),
  dataset(10, "Waiting for a node", "bam", "queued"),
  dataset(11, "Not started yet", "bam", "new"),
  dataset(12, "Paused behind a failed input", "tabular", "paused"),
  dataset(13, "Metadata being set", "vcf", "setting_metadata"),
  dataset(14, "Cancelled", "tabular", "discarded"),
  dataset(15, "An empty result", "txt", "empty"),
  dataset(16, "A deleted intermediate", "bam", "ok", { deleted: true }),
  dataset(17, "A hidden intermediate", "bam", "ok", { visible: false }),
  dataset(18, '<img src=x onerror="alert(1)"> named like markup', "txt", "ok"),
  dataset(19, "", "tabular", "ok"),
  dataset(20, "A dataset with a name long enough that it has to be cut short where the row runs out of room", "tabular", "ok"),
  collection(21, "A collection that failed to fill", "list", 0, "failed"),
  collection(22, "HISAT2 alignments", "list", 6),
  dataset(23, "featureCounts summary", "tabular", "ok"),
  dataset(24, "MultiQC report", "html", "running"),
];

/** The call that opens each card, as a host would hand it over. */
export const openings: Record<string, { tool: string; args: Record<string, unknown> }> = {
  histories: { tool: "show_histories", args: { limit: 10 } },
  history: { tool: "show_history", args: { history_id: "h1", limit: 10, order: "hid-dsc" } },
  datasets: {
    tool: "show_datasets",
    args: { history_id: "h1", dataset_ids: ["d24", "d23", "d7", "not-in-this-history", "d-private"], caption: "The reports from today's run, and the read that failed" },
  },
};

interface Envelope {
  success: boolean;
  data?: unknown;
  message: string;
  count?: number | null;
  pagination?: Record<string, unknown> | null;
}

function page<T>(items: T[], args: Record<string, unknown>) {
  const limit = typeof args.limit === "number" ? args.limit : 20;
  const offset = typeof args.offset === "number" ? args.offset : 0;
  const shown = items.slice(offset, offset + limit);
  const next = offset + shown.length;
  return {
    shown,
    pagination: {
      total_items: items.length,
      returned_items: shown.length,
      limit,
      offset,
      has_next: next < items.length,
      has_previous: offset > 0,
      next_offset: next < items.length ? next : null,
      previous_offset: offset > 0 ? Math.max(0, offset - limit) : null,
    },
  };
}

/** What the server would answer, for the tools a card opens with or pages through. */
export function answer(tool: string, args: Record<string, unknown>): Envelope {
  if (tool === "show_histories" || tool === "get_histories") {
    const named = typeof args.name === "string" ? histories.filter((history) => history.name === args.name) : histories;
    const { shown, pagination } = page(named, args);
    return { success: true, data: shown, message: `Retrieved ${shown.length} histories`, count: shown.length, pagination };
  }
  if (tool === "show_history" || tool === "get_history_contents") {
    if (args.history_id !== "h1") return { success: false, message: `History ID '${String(args.history_id)}' not found.` };
    // As the read does: deleted and hidden items are left out unless they are asked for.
    const kept = contents.filter((item) => item.deleted === (args.deleted === true) && item.visible === (args.visible !== false));
    const ordered = typeof args.order === "string" && args.order.endsWith("-dsc") ? [...kept].reverse() : kept;
    const { shown, pagination } = page(ordered, args);
    return { success: true, data: { history_id: "h1", contents: shown }, message: `Retrieved ${shown.length} items from history`, count: shown.length, pagination };
  }
  if (tool === "show_datasets") {
    const ids = Array.isArray(args.dataset_ids) ? args.dataset_ids.map(String) : [];
    const selected = ids.map((id) => {
      if (id === "d-private") return { id, status: "inaccessible", dataset: { id, hid: 99, name: "Someone else's dataset" } };
      const found = contents.find((item) => item.id === id && item.history_content_type === "dataset");
      return found ? { id, status: "available", dataset: found } : { id, status: "unavailable" };
    });
    const available = selected.filter((entry) => entry.status === "available").length;
    return {
      success: true,
      data: { history_id: args.history_id, selected, available, ...(typeof args.caption === "string" && { caption: args.caption }) },
      message: `Showing ${available} of ${selected.length} selected datasets`,
      count: selected.length,
    };
  }
  return { success: false, message: `The preview has no answer for ${tool}.` };
}
