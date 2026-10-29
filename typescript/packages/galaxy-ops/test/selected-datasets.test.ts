import { describe, expect, it } from "vitest";
import { allOperations, getSelectedDatasets, GalaxyValidationError, MAX_SELECTED_DATASETS } from "../src/index";
import type { GalaxyContext } from "../src/context";
import { mockClient } from "./util/mock-client";

const reads = { name: "reads.fastq", extension: "fastqsanger", state: "ok", deleted: false, visible: true, file_size: 1200, create_time: "2026-10-01T10:00:00", update_time: "2026-10-01T10:05:00" };

/** A Galaxy that answers the contents index with `items`, recording what it was asked. */
function galaxy(items: unknown[] | ((query: any) => unknown[]), status = 200, message = "History not found") {
  const calls: { path: string; params: any }[] = [];
  const ctx = {
    client: mockClient({
      GET: (path, init) => {
        calls.push({ path, params: init?.params });
        if (status !== 200) return { error: { err_msg: message, err_code: status * 1000 + 1 }, response: { status } };
        return { data: typeof items === "function" ? items(init?.params?.query) : items, response: { status: 200 } };
      },
    }),
  } as unknown as GalaxyContext;
  return { ctx, calls };
}

describe("getSelectedDatasets", () => {
  it("asks the contents index for exactly these datasets, with the fields it shows", async () => {
    const { ctx, calls } = galaxy([]);
    await getSelectedDatasets({ historyId: "h1", datasetIds: ["a", "b"] }, ctx);
    expect(calls).toEqual([
      {
        path: "/api/histories/{history_id}/contents",
        params: {
          path: { history_id: "h1" },
          query: {
            v: "dev",
            q: ["type_id-in"],
            qv: ["dataset-a,dataset-b"],
            keys: "id,hid,name,extension,state,deleted,visible,create_time,update_time,file_size,accessible",
          },
        },
      },
    ]);
  });

  it("answers in the order chosen, once per id, and says what became of each", async () => {
    const { ctx } = galaxy([
      { id: "c", hid: 3, history_content_type: "dataset", accessible: false, name: "private.bam", state: "ok" },
      { id: "a", hid: 1, history_content_type: "dataset", accessible: true, ...reads, misc_info: "dropped", peek: "dropped" },
    ]);
    const result = await getSelectedDatasets({ historyId: "h1", datasetIds: ["a", "missing", "c", "a"] }, ctx);
    expect(result).toEqual({
      history_id: "h1",
      available: 1,
      selected: [
        { id: "a", status: "available", dataset: { id: "a", hid: 1, ...reads, accessible: true } },
        { id: "missing", status: "unavailable" },
        { id: "c", status: "inaccessible", dataset: { id: "c", hid: 3, name: "private.bam", state: "ok", accessible: false } },
      ],
    });
  });

  it("does not take a missing accessible flag as access", async () => {
    const { ctx } = galaxy([{ id: "a", hid: 1, ...reads }]);
    const [only] = (await getSelectedDatasets({ historyId: "h1", datasetIds: ["a"] }, ctx)).selected;
    expect(only?.status).toBe("inaccessible");
  });

  it("cuts a very long name instead of dropping the dataset", async () => {
    const { ctx } = galaxy([{ id: "a", hid: 1, accessible: true, name: "x".repeat(500) }]);
    const [only] = (await getSelectedDatasets({ historyId: "h1", datasetIds: ["a"] }, ctx)).selected;
    expect(only?.dataset?.name).toHaveLength(200);
    expect(only?.dataset?.name_truncated).toBe(true);
  });

  it("refuses an empty selection, or more than the limit, before asking Galaxy", async () => {
    const { ctx, calls } = galaxy([]);
    await expect(getSelectedDatasets({ historyId: "h1", datasetIds: [] }, ctx)).rejects.toBeInstanceOf(GalaxyValidationError);
    const tooMany = Array.from({ length: MAX_SELECTED_DATASETS + 1 }, (_, i) => `d${i}`);
    await expect(getSelectedDatasets({ historyId: "h1", datasetIds: tooMany }, ctx)).rejects.toThrow("at most 20 datasets (got 21)");
    expect(calls).toHaveLength(0);
  });

  it("reports a history Galaxy will not read as Galaxy's error", async () => {
    const { ctx } = galaxy([], 404);
    await expect(getSelectedDatasets({ historyId: "nope", datasetIds: ["a"] }, ctx)).rejects.toThrow("History not found");
  });

  it("passes on Galaxy's message for an id it cannot decode", async () => {
    const { ctx } = galaxy([], 400, "Wrong  id ( nope ) specified, unable to decode.");
    await expect(getSelectedDatasets({ historyId: "h1", datasetIds: ["nope"] }, ctx)).rejects.toThrow("unable to decode");
  });

  it("is not an operation, so it adds no tool", () => {
    expect(allOperations.map((op) => op.name)).not.toContain("get_selected_datasets");
  });
});
