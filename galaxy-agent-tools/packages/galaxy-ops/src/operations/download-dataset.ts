// Note: the /api/datasets/{dataset_id}/display endpoint supports `parseAs: "arrayBuffer"` which is
// not reflected in the typed openapi-fetch client. We cast ctx.client.GET to `any` to pass parseAs
// through -- this is intentional and localized to this file.

import { writeFile } from "node:fs/promises";
import { z } from "zod";
import type { GalaxyContext } from "../context";
import { classifyHttp, GalaxyConnectionError } from "../errors";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

export interface DownloadDatasetResult {
  dataset_id: string;
  file_path?: string;
  suggested_filename: string;
  content_available: boolean;
  file_size?: number;
  dataset_info: {
    name?: string;
    extension?: string;
    state?: string;
    genome_build?: string;
    file_size?: number;
  };
}

interface DatasetMeta {
  id?: string;
  name?: string;
  file_ext?: string;
  state?: string;
  genome_build?: string;
  file_size?: number;
  [key: string]: unknown;
}

const input = {
  datasetId: z.string().describe("Dataset id to download"),
  filePath: z.string().optional().describe("Local path to write; omit for in-memory"),
  requireOkState: z.boolean().optional().describe("Throw if dataset state != ok (default true)"),
};

type In = { datasetId: string; filePath?: string; requireOkState?: boolean };

async function run(i: In, ctx: GalaxyContext): Promise<DownloadDatasetResult> {
  // Fetch metadata
  const { data: meta, error: metaError, response: metaResp } = await ctx.client.GET("/api/datasets/{dataset_id}", {
    params: { path: { dataset_id: i.datasetId } },
  });
  if (metaError || !meta) throw classifyHttp(metaResp.status, metaError);

  const m = meta as DatasetMeta;
  const state = m.state;
  const ext = m.file_ext ?? "dat";
  const name = m.name ?? i.datasetId;

  if (i.requireOkState !== false && state !== "ok") {
    throw new GalaxyConnectionError(`dataset ${i.datasetId} not ready (state=${state})`, 409);
  }

  // Fetch content via display endpoint. legacyGet can't be used here because it doesn't support
  // parseAs -- it always parses JSON. The cast to any is intentional and localized to this call.
  // We still throw a typed error on failure so callers get the same error shape as everywhere else.
  const { data: rawData, error: dlError, response: dlResp } = await (ctx.client.GET as any)(
    "/api/datasets/{dataset_id}/display",
    { params: { path: { dataset_id: i.datasetId }, query: { to_ext: ext } }, parseAs: "arrayBuffer" },
  );
  if (dlError || !rawData) throw classifyHttp(dlResp.status, dlError);

  // Build suggested filename
  const dotExt = `.${ext}`;
  const suggested_filename = name.endsWith(dotExt) ? name : `${name}${dotExt}`;

  const dataset_info = {
    name: m.name,
    extension: ext,
    state,
    genome_build: m.genome_build,
    file_size: m.file_size,
  };

  if (i.filePath) {
    await writeFile(i.filePath, Buffer.from(rawData as ArrayBuffer));
    return {
      dataset_id: i.datasetId,
      file_path: i.filePath,
      suggested_filename,
      content_available: false,
      dataset_info,
    };
  }

  return {
    dataset_id: i.datasetId,
    suggested_filename,
    content_available: true,
    file_size: (rawData as ArrayBuffer).byteLength,
    dataset_info,
  };
}

export const downloadDatasetOp: Operation<typeof input, DownloadDatasetResult> = {
  name: "download_dataset",
  domain: "datasets",
  result: {
    kind: "object",
    fields: ["dataset_id", "dataset_info", "suggested_filename", "content_available"],
  },
  // readOnly: false because it can write to disk (filePath branch)
  readOnly: false,
  summary: "Download a dataset's content by id, optionally writing to a local file.",
  input,
  run,
  project: (out, i) => ({
    message: `Dataset ${i.datasetId} (${out.dataset_info.state})${out.file_path ? " -> " + out.file_path : ""}`,
  }),
};

register(downloadDatasetOp as AnyOperation);

export const downloadDataset = (i: In, ctx: GalaxyContext) => runOperation(downloadDatasetOp, i, ctx);
