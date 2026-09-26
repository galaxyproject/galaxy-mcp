import { describe, it, expect } from "vitest";
import { downloadDatasetOp, downloadDataset } from "../../src/operations/download-dataset";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

// filePath branch (writes to disk) is deferred to integration tests.

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("download_dataset", () => {
  it("(a) state ok, no filePath: reports the size and the metadata, and no content", async () => {
    const client = mockClient({
      GET: (path, _init) => {
        if (path.includes("/display")) {
          return {
            data: new Uint8Array([1, 2, 3]).buffer,
            response: { status: 200 },
          };
        }
        // metadata GET
        return {
          data: { id: "d1", name: "my-data", file_ext: "txt", state: "ok", genome_build: "hg19", file_size: 100 },
          response: { status: 200 },
        };
      },
    });
    const out = await downloadDataset({ datasetId: "d1" }, ctxWith(client));
    expect(out.content_available).toBe(true);
    expect(out.suggested_filename).toBe("my-data.txt");
    expect(out.file_size).toBe(3);
    expect(out.dataset_info.state).toBe("ok");
    expect(out.dataset_id).toBe("d1");
    // content_available is about the fetch, not about this object: the three bytes were
    // measured and dropped, and no key here carries them.
    expect(Object.keys(out).sort()).toEqual([
      "content_available",
      "dataset_id",
      "dataset_info",
      "file_size",
      "suggested_filename",
    ]);
  });

  it("(b) state running -> throws with 'not ready' message", async () => {
    const client = mockClient({
      GET: (_path, _init) => {
        return {
          data: { id: "d2", name: "my-data", file_ext: "txt", state: "running", genome_build: "hg19", file_size: 0 },
          response: { status: 200 },
        };
      },
    });
    await expect(downloadDataset({ datasetId: "d2" }, ctxWith(client))).rejects.toThrow(/not ready/);
  });

  it("requireOkState=false bypasses state check", async () => {
    const client = mockClient({
      GET: (path, _init) => {
        if (path.includes("/display")) {
          return { data: new Uint8Array([5, 6]).buffer, response: { status: 200 } };
        }
        return {
          data: { id: "d3", name: "my-data", file_ext: "txt", state: "running", genome_build: null, file_size: 50 },
          response: { status: 200 },
        };
      },
    });
    const out = await downloadDataset({ datasetId: "d3", requireOkState: false }, ctxWith(client));
    expect(out.content_available).toBe(true);
    expect(out.file_size).toBe(2);
  });

  it("declares useDefaultFilename, and changes nothing either way -- as Python does", async () => {
    // Python takes the argument and never reads it: both branches of its body pass the
    // literal use_default_filename=False to bioblend. The name it once stood for comes back
    // as suggested_filename instead, which is what this pins on both settings.
    const meta = { id: "d4", name: "reads", file_ext: "fastq", state: "ok", file_size: 9 };
    const asked: string[] = [];
    const client = mockClient({
      GET: (path) => {
        asked.push(path);
        return path.includes("/display")
          ? { data: new Uint8Array([7]).buffer, response: { status: 200 } }
          : { data: meta, response: { status: 200 } };
      },
    });
    const on = await downloadDataset({ datasetId: "d4", useDefaultFilename: true }, ctxWith(client));
    const off = await downloadDataset({ datasetId: "d4", useDefaultFilename: false }, ctxWith(client));
    expect(on).toEqual(off);
    expect(on.suggested_filename).toBe("reads.fastq");
    expect(on.file_path).toBeUndefined();
    expect(asked).toHaveLength(4);
  });

  it("advertises Python's default for it", () => {
    expect(downloadDatasetOp.input.useDefaultFilename.parse(undefined)).toBe(true);
  });

  it("is not read-only, because filePath overwrites a local file, and says where the bytes go", () => {
    // readOnlyHint means "does not modify its environment", and the filePath branch modifies
    // it. Python's read tag is about the server only, which the summary also states, so the
    // two surfaces disagree on purpose -- see the intentional row in the parity registry.
    expect(downloadDatasetOp.readOnly).toBe(false);
    expect(downloadDatasetOp.summary).toContain("written to that local path");
    expect(downloadDatasetOp.summary).toContain("overwriting what is there");
    expect(downloadDatasetOp.summary).toContain("Nothing on the Galaxy server is changed");
  });

  it("promises no bytes it does not hand back", () => {
    // The no-filePath result is metadata and a byte count, so the summary has to say that
    // filePath is the way to the content and must not offer it any other way.
    expect(downloadDatasetOp.summary).toContain("the only way to get them");
    expect(downloadDatasetOp.summary).toContain("fetched and discarded");
    expect(downloadDatasetOp.summary).not.toMatch(/in memory|in-memory/);
    expect(downloadDatasetOp.input.filePath.description).toContain("no content");
    expect(downloadDatasetOp.input.filePath.description).not.toMatch(/in memory|in-memory/);
  });
});
