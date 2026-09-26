import { z } from "zod";
import type { GetJson } from "../bindings";
import type { GalaxyContext } from "../context";
import { classifyHttp } from "../errors";
import { legacyGet } from "../legacy";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

export type DatasetDetail = GetJson<"/api/datasets/{dataset_id}">;

/** `total_lines` is present only when the whole dataset was read; a window's count is not a file's. */
export interface DatasetPreview {
  lines?: string | null;
  total_lines?: number;
  preview_lines?: number;
  truncated?: boolean;
  error?: string;
}

/** The bytes read, and whether they are the whole dataset or only its head. */
interface Head {
  text: string;
  whole: boolean;
}

// Only the head is fetched; the Python tool downloads the whole file to slice the same lines.
const PREVIEW_BYTES = 256 * 1024;
const DEFAULT_PREVIEW_LINES = 10;

const input = {
  datasetId: z.string().describe("Encoded dataset id"),
  includePreview: z.boolean().default(true).describe("Include a preview of the content (default true)"),
  previewLines: z
    .number()
    .int()
    .default(DEFAULT_PREVIEW_LINES)
    .describe(`Lines of content to preview (default ${DEFAULT_PREVIEW_LINES})`),
};
type In = { datasetId: string; includePreview?: boolean; previewLines?: number };

/** Galaxy's chunked display answer: a line-aligned prefix of a chunkable datatype. */
interface Chunk {
  ck_data?: string;
}

/** The size of a string on the wire, which is what the read window is measured in. */
function byteLength(text: string): number {
  return new TextEncoder().encode(text).byteLength;
}

/** The head of a dataset as text, read without pulling the whole file into memory. */
async function head(ctx: GalaxyContext, datasetId: string): Promise<Head> {
  // The chunked route answers a line-aligned prefix; datatypes it cannot chunk fall through.
  try {
    const chunk = await legacyGet<Chunk>(ctx, "/api/datasets/{dataset_id}/display", {
      params: { path: { dataset_id: datasetId }, query: { offset: 0, ck_size: PREVIEW_BYTES } },
    });
    if (typeof chunk?.ck_data === "string") {
      return { text: chunk.ck_data, whole: byteLength(chunk.ck_data) < PREVIEW_BYTES };
    }
  } catch {
    // not chunkable, or the route answered something else -- read the head instead
  }
  // parseAs is not on the typed client; the cast is localized, as in download_dataset.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const { data, error, response } = await (ctx.client.GET as any)("/api/datasets/{dataset_id}/display", {
    params: { path: { dataset_id: datasetId } },
    parseAs: "arrayBuffer",
  });
  if (error || data == null) throw classifyHttp(response.status, error);
  const all = new Uint8Array(data as ArrayBuffer);
  const whole = all.byteLength <= PREVIEW_BYTES;
  const bytes = whole ? all : all.slice(0, PREVIEW_BYTES);
  const text = new TextDecoder("utf-8", { fatal: false }).decode(bytes);
  // A replacement character means this is not text; the hex sentence is the Python tool's.
  if (text.includes("�")) {
    const hex = Array.from(bytes.slice(0, 100), (b) => b.toString(16).padStart(2, "0")).join("");
    return { text: `[Binary content - first ${Math.min(100, bytes.length)} bytes as hex: ${hex}]`, whole: true };
  }
  return { text, whole };
}

async function preview(ctx: GalaxyContext, datasetId: string, want: number): Promise<DatasetPreview> {
  try {
    const { text, whole } = await head(ctx, datasetId);
    const read = text.split("\n");
    // The window can end mid-line, and half a row is not a row.
    const lines = whole || read.length < 2 ? read : read.slice(0, -1);
    return {
      lines: lines.slice(0, want).join("\n"),
      // Only when the whole dataset was read; past the window the file holds more.
      ...(whole ? { total_lines: lines.length } : {}),
      preview_lines: Math.min(want, lines.length),
      truncated: !whole || lines.length > want,
    };
  } catch (err) {
    // Name an unreadable preview: an absent field reads as a dataset with no content.
    return { error: `Preview unavailable: ${err instanceof Error ? err.message : String(err)}`, lines: null };
  }
}

async function run(i: In, ctx: GalaxyContext): Promise<DatasetDetail> {
  const { data, error, response } = await ctx.client.GET("/api/datasets/{dataset_id}", {
    params: { path: { dataset_id: i.datasetId } },
  });
  if (error || !data) throw classifyHttp(response.status, error);
  const dataset = data as DatasetDetail & { state?: string; preview?: DatasetPreview };
  // Only a dataset in `ok` state has content to read, as on the Python side.
  if ((i.includePreview ?? true) && dataset.state === "ok") {
    return { ...dataset, preview: await preview(ctx, i.datasetId, i.previewLines ?? DEFAULT_PREVIEW_LINES) };
  }
  return dataset;
}

export const getDatasetDetailsOp: Operation<typeof input, DatasetDetail> = {
  name: "get_dataset_details",
  domain: "datasets",
  summary: "Show a dataset's metadata by id (state, extension, name), with a preview of its content.",
  input,
  run,
  project: (d) => ({ message: `Dataset ${(d as { id?: string }).id} state=${(d as { state?: string }).state}` }),
};

register(getDatasetDetailsOp as AnyOperation);

export const getDatasetDetails = (i: In, ctx: GalaxyContext) => runOperation(getDatasetDetailsOp, i, ctx);
