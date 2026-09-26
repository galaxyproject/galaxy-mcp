// Note: the /api/datasets/{dataset_id}/display endpoint supports `parseAs: "arrayBuffer"` which is
// not reflected in the typed openapi-fetch client. We cast ctx.client.GET to `any` to pass parseAs
// through -- this is intentional and localized to this file.

import { writeFile } from "node:fs/promises";
import { z } from "zod";
import type { GalaxyContext } from "../context";
import { classifyHttp, GalaxyConnectionError } from "../errors";
import { register, runOperation } from "./registry";
import type { AnyOperation, InputOf, Operation } from "./types";

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
  filePath: z
    .string()
    .optional()
    .describe(
      "Local path to write the bytes to; omit it and the result is metadata and file_size only, " +
        "with no content",
    ),
  /**
   * Declared because Python declares it, and inert because Python's is inert.
   *
   * Python's `download_dataset` takes the argument and never reads it: both branches of its
   * body pass the literal `use_default_filename=False` to bioblend, so the value a caller
   * sends changes nothing there either. What it once meant survives as `suggested_filename`,
   * which the in-memory branch derives from the dataset's own name and extension and hands
   * back for the caller to use -- on both surfaces. Honouring it here instead would mean
   * writing a file into the caller's working directory that Python would not write, which is
   * a bigger divergence than the one it closes.
   */
  useDefaultFilename: z
    .boolean()
    .default(true)
    .describe(
      "Deprecated - use filePath for specific locations (default: true, ignored when filePath not provided)",
    ),
  requireOkState: z.boolean().default(true).describe("Throw if dataset state != ok (default true)"),
};

type In = {
  datasetId: string;
  filePath?: string;
  useDefaultFilename?: boolean;
  requireOkState?: boolean;
};

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

  // No filePath, so the buffer is measured and dropped -- `content_available` says the fetch
  // worked, not that the bytes are in here. Python answers the same way (its in-memory branch
  // returns the length and not the content), so filePath is the only route to the bytes on
  // either surface.
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
  /**
   * Not read-only, because `readOnlyHint` is about the environment, not about Galaxy.
   *
   * Nothing here creates, changes or deletes anything on the SERVER -- it is two GETs, and
   * Python's `read` tag says that much, since the tags only gate
   * GALAXY_MCP_INCLUDE/EXCLUDE_TAGS. The MCP hint is not scoped that way: the SDK defines it
   * as a tool that "does not modify its environment", and with `filePath` this one overwrites
   * whatever file is at that path. Clients gate approval on the hint, so a true here would be
   * a false annotation -- and the description disclosing the overwrite does not correct it.
   * Dannon's call (2026-09-26): the hint stays false and the registry records the mismatch
   * with Python's tag as intentional, because each flag is right about its own scope.
   */
  readOnly: false,
  summary:
    "Download a dataset's content by id. With filePath the bytes are written to that local " +
    "path, overwriting what is there, and that is the only way to get them. Without filePath " +
    "the content is fetched and discarded: the result is metadata only -- file_size, " +
    "content_available, and the dataset's name, extension, state and genome build -- with no " +
    "content in it. Nothing on the Galaxy server is changed either way.",
  input,
  run,
  project: (out, i) => ({
    message: `Dataset ${i.datasetId} (${out.dataset_info.state})${out.file_path ? " -> " + out.file_path : ""}`,
  }),
};

register(downloadDatasetOp as AnyOperation);

export const downloadDataset = (i: In, ctx: GalaxyContext) => runOperation(downloadDatasetOp, i as InputOf<typeof input>, ctx);
