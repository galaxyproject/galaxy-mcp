import { describe, it, expect, vi, beforeEach } from "vitest";
import { uploadFile, uploadFileOp } from "../../src/operations/upload-file";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

// Mock tus so no real network runs in unit tests.
vi.mock("../../src/tus-upload", () => ({
  tusUploadFile: vi.fn().mockResolvedValue("fake-session-id"),
}));

// existsSync is also mocked so tests don't require real files on disk.
vi.mock("node:fs", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:fs")>();
  return { ...actual, existsSync: vi.fn().mockReturnValue(true) };
});

const ctxWith = (client: any): GalaxyContext => ({
  client,
  poll: DEFAULT_POLL,
  baseUrl: "https://galaxy.example",
  apiKey: "test-key",
});

describe("upload_file", () => {
  let capturedBody: unknown;

  beforeEach(async () => {
    capturedBody = undefined;
    const { tusUploadFile } = await import("../../src/tus-upload");
    vi.mocked(tusUploadFile).mockClear();
  });

  it("(a) posts correct Fetch API body shape", async () => {
    const client = mockClient({
      POST: (_path, init) => {
        capturedBody = (init as any)?.body;
        return { data: { outputs: [{ id: "ds1" }], jobs: [{ id: "j1" }] }, response: { status: 200 } };
      },
    });
    const out = await uploadFile({ path: "/tmp/reads.fastq.gz", historyId: "h1" }, ctxWith(client));
    expect(out.outputs).toHaveLength(1);
    expect(out.jobs).toHaveLength(1);

    const body = capturedBody as Record<string, unknown>;
    expect(body["history_id"]).toBe("h1");
    expect((body["files_0|file_data"] as any).session_id).toBe("fake-session-id");
    expect((body["files_0|file_data"] as any).name).toBe("reads.fastq.gz");

    const targets = body["targets"] as any[];
    expect(targets[0].destination.type).toBe("hdas");
    expect(targets[0].elements[0].name).toBe("reads.fastq.gz");
    expect(targets[0].elements[0].src).toBe("files");
    expect(body["auto_decompress"]).toBe(false);
  });

  it("(b) omits history_id from body when not provided", async () => {
    const client = mockClient({
      POST: (_path, init) => {
        capturedBody = (init as any)?.body;
        return { data: { outputs: [], jobs: [] }, response: { status: 200 } };
      },
    });
    await uploadFile({ path: "/tmp/data.tsv" }, ctxWith(client));
    const body = capturedBody as Record<string, unknown>;
    expect("history_id" in body).toBe(false);
  });

  it("(c) throws file-not-found without calling tus or POST", async () => {
    const { existsSync } = await import("node:fs");
    vi.mocked(existsSync).mockReturnValueOnce(false);

    const { tusUploadFile } = await import("../../src/tus-upload");
    const tusMock = vi.mocked(tusUploadFile);

    const client = mockClient({
      POST: () => ({ data: {}, response: { status: 200 } }),
    });
    await expect(uploadFile({ path: "/nonexistent/file.txt" }, ctxWith(client))).rejects.toThrow(
      /file not found/,
    );
    expect(tusMock).not.toHaveBeenCalled();
  });

  it("(e) throws GalaxyConnectionError when apiKey is missing without calling tus", async () => {
    const { tusUploadFile } = await import("../../src/tus-upload");
    const tusMock = vi.mocked(tusUploadFile);

    const client = mockClient({
      POST: () => ({ data: {}, response: { status: 200 } }),
    });
    const ctxNoKey: GalaxyContext = { client, poll: DEFAULT_POLL, baseUrl: "https://galaxy.example" };
    await expect(uploadFile({ path: "/tmp/reads.fastq.gz" }, ctxNoKey)).rejects.toThrow(
      /baseUrl and apiKey are required for file upload/,
    );
    expect(tusMock).not.toHaveBeenCalled();
  });

  it("(d) project() includes basename and job count", () => {
    const out = { outputs: [{}], jobs: [{}], extra: "x" };
    const meta = uploadFileOp.project!(out, { path: "/some/dir/myfile.bam" });
    expect(meta.message).toContain("myfile.bam");
    expect(meta.message).toContain("1 job(s)");
  });
});
