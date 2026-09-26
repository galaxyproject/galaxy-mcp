import { describe, it, expect } from "vitest";
import { getDatasetDetails } from "../../src/operations/get-dataset-details";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

const bytes = (text: string) => new TextEncoder().encode(text).buffer;

/** A dataset read, and its content read from the display route.
 *
 * `chunked` stands for a datatype Galaxy can chunk (the preview's first choice); `content`
 * for one it cannot, which falls back to reading the head of the stream.
 */
function client(
  meta: Record<string, unknown>,
  opts: { chunked?: string; content?: ArrayBuffer; seen?: any[] } = {},
) {
  return mockClient({
    GET: (path: string, init: any) => {
      if (path === "/api/datasets/{dataset_id}") return { data: meta, response: { status: 200 } };
      expect(path).toBe("/api/datasets/{dataset_id}/display");
      opts.seen?.push(init);
      const asksForAChunk = init?.params?.query?.ck_size !== undefined;
      if (asksForAChunk) {
        if (opts.chunked === undefined) return { error: { err_msg: "not chunkable" }, response: { status: 400 } };
        return { data: { ck_data: opts.chunked }, response: { status: 200 } };
      }
      if (opts.content === undefined) return { error: { err_msg: "not ready" }, response: { status: 400 } };
      return { data: opts.content, response: { status: 200 } };
    },
  });
}

describe("get_dataset_details", () => {
  it("shows a dataset by id", async () => {
    const out = await getDatasetDetails({ datasetId: "d1", includePreview: false }, ctxWith(client({ id: "d1" })));
    expect((out as any).id).toBe("d1");
    expect(out).not.toHaveProperty("preview");
  });

  it("previews the head of the content, saying how much it left out", async () => {
    const c = client({ id: "d1", state: "ok" }, { chunked: "a\nb\nc\nd" });
    const out: any = await getDatasetDetails({ datasetId: "d1", previewLines: 2 }, ctxWith(c));
    expect(out.preview).toEqual({ lines: "a\nb", total_lines: 4, preview_lines: 2, truncated: true });
  });

  it("previews by default, and only for a dataset that has content", async () => {
    const ok: any = await getDatasetDetails({ datasetId: "d1" }, ctxWith(client({ id: "d1", state: "ok" }, { chunked: "x" })));
    expect(ok.preview.lines).toBe("x");
    const running: any = await getDatasetDetails({ datasetId: "d1" }, ctxWith(client({ id: "d1", state: "running" })));
    expect(running).not.toHaveProperty("preview");
  });

  it("names a preview it could not read rather than dropping the field", async () => {
    const out: any = await getDatasetDetails({ datasetId: "d1" }, ctxWith(client({ id: "d1", state: "ok" })));
    expect(out.preview.lines).toBeNull();
    expect(out.preview.error).toMatch(/Preview unavailable/);
  });

  it("reports binary content as hex instead of mojibake", async () => {
    const c = client({ id: "d1", state: "ok" }, { content: new Uint8Array([0xff, 0xfe, 0x00, 0x01]).buffer });
    const out: any = await getDatasetDetails({ datasetId: "d1" }, ctxWith(c));
    expect(out.preview.lines).toMatch(/^\[Binary content - first 4 bytes as hex: fffe0001\]/);
  });

  it("states a line count only when it read the whole dataset", async () => {
    // A window is not a file: counting the lines in 256 KB of a gigabyte would understate it
    // by orders of magnitude, so the count is withheld and truncation is reported instead.
    const filled = "row\n".repeat(64 * 1024); // 256 KB exactly, so the read hit its window
    const c = client({ id: "d1", state: "ok" }, { chunked: filled });
    const out: any = await getDatasetDetails({ datasetId: "d1", previewLines: 2 }, ctxWith(c));
    expect(out.preview).not.toHaveProperty("total_lines");
    expect(out.preview.truncated).toBe(true);
    expect(out.preview.lines).toBe("row\nrow");
  });

  it("drops a trailing line the window may have cut in half", async () => {
    const content = "a\n".repeat(128 * 1024) + "partial"; // over the window, ends mid-row
    const c = client({ id: "d1", state: "ok" }, { content: bytes(content) });
    const out: any = await getDatasetDetails({ datasetId: "d1", previewLines: 2 }, ctxWith(c));
    expect(out.preview).not.toHaveProperty("total_lines");
    expect(out.preview.lines).toBe("a\na");
  });

  it("asks for a line-aligned chunk rather than streaming the whole dataset", async () => {
    const seen: any[] = [];
    const c = client({ id: "d1", state: "ok" }, { chunked: "a\nb", seen });
    await getDatasetDetails({ datasetId: "d1" }, ctxWith(c));
    expect(seen).toHaveLength(1);
    expect(seen[0].params.query.ck_size).toBe(256 * 1024);
    expect(seen[0].parseAs).toBeUndefined();
  });

  it("falls back to the head of the stream for a datatype Galaxy cannot chunk", async () => {
    const seen: any[] = [];
    const c = client({ id: "d1", state: "ok" }, { content: bytes("p\nq\nr"), seen });
    const out: any = await getDatasetDetails({ datasetId: "d1", previewLines: 2 }, ctxWith(c));
    expect(out.preview.lines).toBe("p\nq");
    expect(seen.map((s) => s.params.query?.ck_size !== undefined)).toEqual([true, false]);
  });
});
