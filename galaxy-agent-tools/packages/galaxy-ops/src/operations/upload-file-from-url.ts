/**
 * upload_file_from_url -- classic upload1 tool form via /api/tools.
 *
 * Mirrors BioBlend's put_url / paste_content which POSTs to the classic Galaxy
 * upload controller. Returns the run result immediately with no wait.
 */

import { z } from "zod";
import type { GalaxyContext } from "../context";
import { legacyPost } from "../legacy";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

export interface UploadFileFromUrlResult {
  outputs?: unknown[];
  jobs?: unknown[];
  [k: string]: unknown;
}

const input = {
  url: z.string().describe("URL of the file to upload"),
  historyId: z.string().optional().describe("Target history id; omit to use Galaxy's default"),
  fileType: z.string().optional().describe("Galaxy file type, e.g. 'fasta', 'fastq', 'bam' (default: 'auto')"),
  dbkey: z.string().optional().describe("Genome build / database key (default: '?')"),
  fileName: z.string().optional().describe("Name for the uploaded dataset in Galaxy (inferred from URL if omitted)"),
};

type In = { url: string; historyId?: string; fileType?: string; dbkey?: string; fileName?: string };

async function run(i: In, ctx: GalaxyContext): Promise<UploadFileFromUrlResult> {
  const toolInputs: Record<string, unknown> = {
    file_type: i.fileType ?? "auto",
    dbkey: i.dbkey ?? "?",
    "files_0|type": "upload_dataset",
  };
  if (i.fileName) {
    toolInputs["files_0|NAME"] = i.fileName;
  }

  const body: Record<string, unknown> = {
    tool_id: "upload1",
    inputs: toolInputs,
    "files_0|url_paste": i.url,
  };
  if (i.historyId) {
    body["history_id"] = i.historyId;
  }

  return legacyPost<UploadFileFromUrlResult>(ctx, "/api/tools", { body });
}

export const uploadFileFromUrlOp: Operation<typeof input, UploadFileFromUrlResult> = {
  name: "upload_file_from_url",
  domain: "datasets",
  readOnly: false,
  summary:
    "Upload a file from a URL to a Galaxy history via the classic upload1 tool form. Returns immediately with job info (no wait).",
  input,
  run,
  project: (_out, i) => ({
    message: `Submitted URL upload to history ${i.historyId ?? "(default)"}`,
  }),
};

register(uploadFileFromUrlOp as AnyOperation);

export const uploadFileFromUrl = (i: In, ctx: GalaxyContext) => runOperation(uploadFileFromUrlOp, i, ctx);
