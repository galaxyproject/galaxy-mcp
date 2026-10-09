/**
 * The preview cases are Python's, case for case
 * (`python/tests/test_dataset_operations.py`), so the two surfaces answer the
 * same thing for the same dataset: the same key set, the same line slice, the same route.
 */
import { describe, it, expect } from "vitest";
import {
  getDatasetDetailsOp,
  getDatasetDetails,
  type DatasetPreview,
} from "../../src/operations/get-dataset-details";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

const SHOW = "/api/datasets/{dataset_id}";
const TEXT = "/api/datasets/{dataset_id}/get_content_as_text";

/** Galaxy's answer to the two routes this op uses, recording which were asked for. */
function datasetClient(
  meta: Record<string, unknown>,
  // `item_data` is unknown rather than `string | null`, so a case can hand back the 200
  // Galaxy's schema says it will not send and this op still has to survive. `body` goes one
  // further and replaces the whole reply, for a 200 that is not even an object.
  text?: { item_data: unknown; truncated?: boolean } | { throws: unknown } | { body: unknown },
  asked: string[] = [],
) {
  return mockClient({
    GET: (path, init) => {
      asked.push(path);
      if (path === SHOW) {
        expect(init.params.path.dataset_id).toBe(meta.id);
        return { data: meta, response: { status: 200 } };
      }
      if (path === TEXT) {
        if (!text) throw new Error("the text route must not be asked for here");
        if ("throws" in text) return { error: text.throws, response: { status: 500 } };
        if ("body" in text) return { data: text.body, response: { status: 200 } };
        return {
          data: { item_data: text.item_data, truncated: text.truncated ?? false, item_url: "/datasets/x/display" },
          response: { status: 200 },
        };
      }
      throw new Error(`unexpected path ${path}`);
    },
  });
}

const previewOf = (out: unknown) => (out as { preview?: DatasetPreview }).preview;

/** The dataset record, which travels under `dataset` rather than at the top level. */
const recordOf = (out: unknown) => (out as { dataset: Record<string, unknown> }).dataset;

describe("get_dataset_details", () => {
  it("shows a dataset by id", async () => {
    const out = await getDatasetDetails(
      { datasetId: "d1", includePreview: false },
      ctxWith(datasetClient({ id: "d1", state: "ok", file_ext: "txt" })),
    );
    expect(recordOf(out).id).toBe("d1");
    expect((out as any).dataset_id).toBe("d1");
  });

  it("slices the peek to previewLines and reports both truncation facts", async () => {
    const out = await getDatasetDetails(
      { datasetId: "dataset123", includePreview: true, previewLines: 3 },
      ctxWith(
        datasetClient(
          { id: "dataset123", name: "test_data.txt", state: "ok", extension: "txt", file_size: 1024 },
          { item_data: "line1\nline2\nline3\nline4\nline5\n" },
        ),
      ),
    );
    expect(recordOf(out).name).toBe("test_data.txt");
    const preview = previewOf(out)!;
    expect(preview.lines).toBe("line1\nline2\nline3");
    expect(preview.preview_lines).toBe(3);
    expect(preview.truncated).toBe(true);
    expect(preview.content_truncated_by_galaxy).toBe(false);
    // No total: what came back is a peek, and a line count taken from it reads as a count
    // of the dataset.
    expect(Object.keys(preview).sort()).toEqual([
      "content_truncated_by_galaxy",
      "lines",
      "preview_lines",
      "truncated",
    ]);
  });

  it("asks the bounded text route, never the dataset", async () => {
    const asked: string[] = [];
    await getDatasetDetails(
      { datasetId: "dataset123" },
      ctxWith(
        datasetClient(
          { id: "dataset123", name: "test_data.txt", state: "ok" },
          { item_data: "a\nb\n" },
          asked,
        ),
      ),
    );
    expect(asked).toEqual([SHOW, TEXT]);
  });

  it("keeps Galaxy's own truncation flag apart from our line slice", async () => {
    const out = await getDatasetDetails(
      { datasetId: "dataset123", includePreview: true, previewLines: 10 },
      ctxWith(
        datasetClient(
          { id: "dataset123", name: "big.txt", state: "ok" },
          { item_data: "a\nb\n", truncated: true },
        ),
      ),
    );
    const preview = previewOf(out)!;
    expect(preview.content_truncated_by_galaxy).toBe(true);
    // Our own line slice did not cut anything; Galaxy's byte cap did.
    expect(preview.truncated).toBe(false);
    expect(preview.lines).toBe("a\nb\n");
  });

  it("leaves the metadata usable when the preview cannot be fetched", async () => {
    const out = await getDatasetDetails(
      { datasetId: "dataset123" },
      ctxWith(
        datasetClient(
          { id: "dataset123", name: "test_data.txt", state: "ok" },
          { throws: { err_msg: "boom" } },
        ),
      ),
    );
    expect(recordOf(out).name).toBe("test_data.txt");
    const preview = previewOf(out)!;
    expect(preview.error).toContain("Preview unavailable");
    expect(preview.lines).toBeNull();
    expect(Object.keys(preview).sort()).toEqual(["error", "lines"]);
  });

  it("keeps the metadata when item_data comes back as something other than text", async () => {
    // Galaxy's schema says string or null, so this 200 should not happen -- and if it does,
    // Python hits AttributeError on .split() inside its try and answers with the preview
    // error while keeping the metadata. Anything else loses a call the caller can use.
    const out = await getDatasetDetails(
      { datasetId: "dataset123" },
      ctxWith(
        datasetClient({ id: "dataset123", name: "test_data.txt", state: "ok" }, { item_data: 42 }),
      ),
    );
    expect(recordOf(out).name).toBe("test_data.txt");
    const preview = previewOf(out)!;
    expect(preview.lines).toBeNull();
    expect(preview.error).toContain("Preview unavailable");
    expect(Object.keys(preview).sort()).toEqual(["error", "lines"]);
  });

  it.each([
    ["an array", [] as unknown],
    ["a string", "line1\nline2" as unknown],
    ["null", null as unknown],
    ["a number", 7 as unknown],
  ])("calls a 200 whose JSON is %s a failed preview, not an unpreviewable datatype", async (_kind, body) => {
    // Python reads the reply with `body.get("item_data")`, which raises AttributeError for
    // every one of these and lands in its own except: {error: "Preview unavailable: ...",
    // lines: None}. Reading item_data off them here would find undefined and blame the
    // datatype, which is a different diagnosis for the same malformed reply.
    const out = await getDatasetDetails(
      { datasetId: "d1" },
      ctxWith(datasetClient({ id: "d1", name: "test_data.txt", state: "ok" }, { body })),
    );
    expect(recordOf(out).name).toBe("test_data.txt");
    const preview = previewOf(out)!;
    expect(preview.lines).toBeNull();
    expect(preview.error).toContain("Preview unavailable");
    expect(preview.error).not.toContain("text datatypes only");
    expect(Object.keys(preview).sort()).toEqual(["error", "lines"]);
  });

  it("answers a fractional previewLines the way Python's slice does", async () => {
    // MCP and the CLI reject it at the schema (pinned below), so this is the direct library
    // call: Python's `lines[:2.5]` raises TypeError inside its try and comes back as the
    // preview error with the metadata intact, and so does this.
    const out = await getDatasetDetails(
      { datasetId: "d1", previewLines: 2.5 },
      ctxWith(datasetClient({ id: "d1", name: "test_data.txt", state: "ok" }, { item_data: "a\nb\nc" })),
    );
    expect(recordOf(out).name).toBe("test_data.txt");
    const preview = previewOf(out)!;
    expect(preview.lines).toBeNull();
    expect(preview.error).toContain("Preview unavailable");
    expect(Object.keys(preview).sort()).toEqual(["error", "lines"]);
  });

  it("slices a negative previewLines from the end, as Python does, without an error", async () => {
    // Not symmetric with the fractional case on purpose: `lines[:-2]` is valid Python and
    // `slice(0, -2)` drops the same two lines, so neither surface calls it a failure.
    const out = await getDatasetDetails(
      { datasetId: "d1", previewLines: -2 },
      ctxWith(datasetClient({ id: "d1", state: "ok" }, { item_data: "a\nb\nc\nd" })),
    );
    const preview = previewOf(out)!;
    expect(preview.lines).toBe("a\nb");
    expect(preview.preview_lines).toBe(-2);
    expect(preview.truncated).toBe(true);
  });

  it("says so rather than guessing when the datatype has no text preview", async () => {
    const out = await getDatasetDetails(
      { datasetId: "dataset123" },
      ctxWith(
        // hda_manager.text_data returns nothing at all for a datatype that is not text, so
        // Galaxy answers with item_data null rather than bytes to guess at.
        datasetClient({ id: "dataset123", name: "test_data.bin", state: "ok" }, { item_data: null }),
      ),
    );
    const preview = previewOf(out)!;
    expect(preview.lines).toBeNull();
    expect(preview.error).toContain("text datatypes only");
    expect(Object.keys(preview).sort()).toEqual(["content_truncated_by_galaxy", "error", "lines"]);
  });

  it("answers the no-text-preview case without ever reaching previewLines", async () => {
    // Python's null branch returns before the slice, so a preview_lines it would otherwise
    // choke on never comes up. The order of the two checks here is that order.
    const out = await getDatasetDetails(
      { datasetId: "d1", previewLines: 2.5 },
      ctxWith(datasetClient({ id: "d1", state: "ok" }, { item_data: null, truncated: true })),
    );
    const preview = previewOf(out)!;
    expect(preview.error).toContain("text datatypes only");
    expect(preview.content_truncated_by_galaxy).toBe(true);
  });

  it("asks for no preview when includePreview is false", async () => {
    const asked: string[] = [];
    const out = await getDatasetDetails(
      { datasetId: "dataset123", includePreview: false },
      ctxWith(datasetClient({ id: "dataset123", name: "test_data.txt", state: "ok" }, undefined, asked)),
    );
    expect(previewOf(out)).toBeUndefined();
    expect(asked).toEqual([SHOW]);
  });

  it("asks for no preview for a dataset that is not ok, whatever was requested", async () => {
    const asked: string[] = [];
    const out = await getDatasetDetails(
      { datasetId: "dataset123", includePreview: true },
      ctxWith(datasetClient({ id: "dataset123", name: "half.txt", state: "running" }, undefined, asked)),
    );
    expect(previewOf(out)).toBeUndefined();
    expect(asked).toEqual([SHOW]);
  });

  it("previews ten lines by default, as Python does", async () => {
    const twelve = Array.from({ length: 12 }, (_, n) => `line${n + 1}`).join("\n");
    const out = await getDatasetDetails(
      { datasetId: "d1" },
      ctxWith(datasetClient({ id: "d1", state: "ok" }, { item_data: twelve })),
    );
    const preview = previewOf(out)!;
    expect(preview.preview_lines).toBe(10);
    expect(preview.truncated).toBe(true);
    expect(preview.lines?.split("\n")).toHaveLength(10);
  });

  it("advertises Python's two preview parameters with Python's defaults", () => {
    const shape = getDatasetDetailsOp.input;
    expect(shape.includePreview.parse(undefined)).toBe(true);
    expect(shape.previewLines.parse(undefined)).toBe(10);
    expect(shape.previewLines.safeParse(2.5).success).toBe(false);
  });
});
