/**
 * Thin wrapper around tus-js-client for uploading a local file to the Galaxy
 * resumable-upload endpoint. Isolated here so op tests can vi.mock this module
 * without hitting real network.
 *
 * The session_id is the final path segment of the URL the tus server assigns
 * (same derivation as BioBlend's _tus_uploader_session_id monkeypatch).
 */

import { readFileSync, statSync } from "node:fs";
import { basename } from "node:path";
import { Upload } from "tus-js-client";

export interface TusUploadOptions {
  baseUrl: string;
  apiKey: string;
  path: string;
  signal?: AbortSignal;
}

export function tusUploadFile(opts: TusUploadOptions): Promise<string> {
  return new Promise((resolve, reject) => {
    const { baseUrl, apiKey, path, signal } = opts;
    const file = basename(path);
    const data = readFileSync(path);
    const size = statSync(path).size;

    const upload = new Upload(data, {
      endpoint: `${baseUrl}/api/upload/resumable_upload`,
      headers: { "x-api-key": apiKey },
      metadata: { filename: file, filetype: "application/octet-stream" },
      uploadSize: size,
      retryDelays: [0, 1000, 3000],
      onSuccess() {
        // upload.url is the canonical tus upload location set from the creation 201 Location.
        const sessionId = (upload.url ?? "").split("/").pop() ?? "";
        if (!sessionId) {
          reject(new Error("tus upload succeeded but session_id could not be derived from upload URL"));
          return;
        }
        resolve(sessionId);
      },
      onError(err) {
        reject(err);
      },
    });

    // Wire AbortSignal -> tus abort
    if (signal) {
      if (signal.aborted) {
        upload.abort(false).catch(() => {});
        reject(new DOMException("Aborted", "AbortError"));
        return;
      }
      signal.addEventListener("abort", () => {
        upload.abort(false).catch(() => {});
        reject(new DOMException("Aborted", "AbortError"));
      }, { once: true });
    }

    upload.start();
  });
}
