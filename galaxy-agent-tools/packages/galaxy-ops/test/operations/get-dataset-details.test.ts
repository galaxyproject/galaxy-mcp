import { describe, it, expect } from "vitest";
import { getDatasetDetailsOp, getDatasetDetails } from "../../src/operations/get-dataset-details";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("get_dataset_details", () => {
  it("shows a dataset by id", async () => {
    const client = mockClient({
      GET: (path, init) => {
        expect(path).toBe("/api/datasets/{dataset_id}");
        expect(init.params.path.dataset_id).toBe("d1");
        return { data: { id: "d1", state: "ok", file_ext: "txt" }, response: { status: 200 } };
      },
    });
    const out = await getDatasetDetails({ datasetId: "d1" }, ctxWith(client));
    expect((out as any).id).toBe("d1");
  });
});
