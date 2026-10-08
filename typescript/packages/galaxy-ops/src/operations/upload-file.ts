/**
 * upload_file -- tus resumable upload + Fetch API finalize.
 *
 * Local-fs op: works in stdio/CLI deployments; diverges for hosted MCP where
 * the server process can't access the caller's filesystem. Document at
 * integration boundary.
 *
 * Note: a real end-to-end tus round-trip is a gated live integration (deferred).
 * The unit test mocks tusUploadFile so no network runs in CI.
 */

import { existsSync } from "node:fs";
import { basename } from "node:path";
import { z } from "zod";
import type { GalaxyContext } from "../context";
import { GalaxyConnectionError } from "../errors";
import { legacyPost } from "../legacy";
import { tusUploadFile } from "../tus-upload";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

export interface UploadFileResult {
  outputs?: unknown[];
  jobs?: unknown[];
  [k: string]: unknown;
}

const input = {
  path: z.string().describe("Local file path to upload"),
  historyId: z.string().optional().describe("Target history id; omit to use Galaxy's default"),
};

type In = { path: string; historyId?: string };

async function run(i: In, ctx: GalaxyContext): Promise<UploadFileResult> {
  if (!existsSync(i.path)) {
    throw new GalaxyConnectionError(`file not found: ${i.path}`, 400);
  }

  if (!ctx.baseUrl || !ctx.apiKey) {
    throw new GalaxyConnectionError("baseUrl and apiKey are required for file upload", 400);
  }

  const file = basename(i.path);
  const baseUrl = ctx.baseUrl;
  const apiKey = ctx.apiKey;

  const sessionId = await tusUploadFile({ baseUrl, apiKey, path: i.path, signal: ctx.signal });

  const payload: Record<string, unknown> = {
    targets: [
      {
        destination: { type: "hdas" },
        elements: [
          {
            src: "files",
            ext: "auto",
            dbkey: "?",
            to_posix_lines: true,
            space_to_tab: false,
            name: file,
          },
        ],
      },
    ],
    "files_0|file_data": { session_id: sessionId, name: file },
    auto_decompress: false,
  };
  if (i.historyId) {
    payload["history_id"] = i.historyId;
  }

  return legacyPost<UploadFileResult>(ctx, "/api/tools/fetch", { body: payload });
}

export const uploadFileOp: Operation<typeof input, UploadFileResult> = {
  name: "upload_file",
  domain: "datasets",
  readOnly: false,
  summary:
    "Upload a local file to a Galaxy history via the tus resumable-upload protocol, then finalize via the Fetch API. Returns immediately with job info (no wait).",
  input,
  run,
  project: (out, i) => ({
    message: `Uploaded ${basename(i.path)} (${out.jobs?.length ?? 0} job(s))`,
  }),
};

register(uploadFileOp as AnyOperation);

export const uploadFile = (i: In, ctx: GalaxyContext) => runOperation(uploadFileOp, i, ctx);
