import { describe, it, expect, vi, beforeEach } from "vitest";
import { uploadFileFromUrl, uploadFileFromUrlOp } from "../../src/operations/upload-file-from-url";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("upload_file_from_url", () => {
  let capturedBody: unknown;

  beforeEach(() => {
    capturedBody = undefined;
  });

  it("(a) posts correct body with defaults", async () => {
    const client = mockClient({
      POST: (_path, init) => {
        capturedBody = (init as any)?.body;
        return { data: { outputs: [{ id: "ds1" }], jobs: [{ id: "j1" }] }, response: { status: 200 } };
      },
    });
    const out = await uploadFileFromUrl(
      { url: "https://example.com/data.fasta", historyId: "hist1" },
      ctxWith(client),
    );
    expect(out.outputs).toHaveLength(1);

    const body = capturedBody as Record<string, unknown>;
    expect(body["tool_id"]).toBe("upload1");
    expect(body["history_id"]).toBe("hist1");
    expect(body["files_0|url_paste"]).toBe("https://example.com/data.fasta");

    const inputs = body["inputs"] as Record<string, unknown>;
    expect(inputs["file_type"]).toBe("auto");
    expect(inputs["dbkey"]).toBe("?");
    expect(inputs["files_0|type"]).toBe("upload_dataset");
    // fileName not provided -> files_0|NAME must be absent
    expect("files_0|NAME" in inputs).toBe(false);
  });

  it("(b) respects fileType and dbkey overrides", async () => {
    const client = mockClient({
      POST: (_path, init) => {
        capturedBody = (init as any)?.body;
        return { data: { outputs: [], jobs: [] }, response: { status: 200 } };
      },
    });
    await uploadFileFromUrl(
      { url: "https://example.com/ref.fa", fileType: "fasta", dbkey: "hg38" },
      ctxWith(client),
    );
    const body = capturedBody as Record<string, unknown>;
    const inputs = body["inputs"] as Record<string, unknown>;
    expect(inputs["file_type"]).toBe("fasta");
    expect(inputs["dbkey"]).toBe("hg38");
  });

  it("(c) includes files_0|NAME only when fileName is given", async () => {
    const client = mockClient({
      POST: (_path, init) => {
        capturedBody = (init as any)?.body;
        return { data: { outputs: [], jobs: [] }, response: { status: 200 } };
      },
    });
    await uploadFileFromUrl(
      { url: "https://example.com/mystery.bin", fileName: "named.tsv" },
      ctxWith(client),
    );
    const body = capturedBody as Record<string, unknown>;
    const inputs = body["inputs"] as Record<string, unknown>;
    expect(inputs["files_0|NAME"]).toBe("named.tsv");
    // history_id omitted
    expect("history_id" in body).toBe(false);
  });

  it("(d) project() message references historyId", () => {
    const meta = uploadFileFromUrlOp.project!({}, { url: "https://x.com/f.fa", historyId: "h42" });
    expect(meta.message).toContain("h42");
  });

  it("(e) project() message uses '(default)' when historyId absent", () => {
    const meta = uploadFileFromUrlOp.project!({}, { url: "https://x.com/f.fa" });
    expect(meta.message).toContain("(default)");
  });
});
