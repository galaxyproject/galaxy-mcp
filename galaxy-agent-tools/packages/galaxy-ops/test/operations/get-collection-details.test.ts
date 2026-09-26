import { describe, it, expect } from "vitest";
import { getCollectionDetails, getCollectionDetailsOp } from "../../src/operations/get-collection-details";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import { GalaxyNotFoundError } from "../../src/errors";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

const element = (n: number) => ({
  element_identifier: `sample${n}`,
  element_type: "hda",
  object: { id: `d${n}`, name: `data ${n}`, state: "ok", extension: "fastqsanger", file_size: 10 * n },
});

const collection = (count: number) => ({
  id: "c1",
  name: "reads",
  collection_type: "list",
  element_count: count,
  populated: true,
  state: "ok",
  elements: Array.from({ length: count }, (_, k) => element(k + 1)),
});

const clientFor = (count: number) =>
  mockClient({ GET: () => ({ data: collection(count), response: { status: 200 } }) });

describe("get_collection_details", () => {
  it("reports the collection's own metadata apart from its members", async () => {
    const out = await getCollectionDetails({ collectionId: "c1" }, ctxWith(clientFor(2)));
    expect(out.collection).toEqual({
      id: "c1",
      name: "reads",
      collection_type: "list",
      element_count: 2,
      populated: true,
      state: "ok",
    });
    expect(out.collection_id).toBe("c1");
    expect(out.history_content_type).toBe("dataset_collection");
  });

  it("flattens each member with the dataset it points at", async () => {
    const out = await getCollectionDetails({ collectionId: "c1" }, ctxWith(clientFor(1)));
    expect(out.elements).toEqual([
      {
        element_index: 0,
        element_identifier: "sample1",
        element_type: "hda",
        object_id: "d1",
        name: "data 1",
        state: "ok",
        extension: "fastqsanger",
        file_size: 10,
      },
    ]);
  });

  it("names the follow-up call, because an element carries an id and not its content", async () => {
    const out = await getCollectionDetails({ collectionId: "c1" }, ctxWith(clientFor(1)));
    expect(out.note).toMatch(/get_dataset_details\(object_id\)/);
  });

  it("says when it truncated, and keeps the real element count", async () => {
    const out = await getCollectionDetails({ collectionId: "c1", maxElements: 2 }, ctxWith(clientFor(5)));
    expect(out.elements).toHaveLength(2);
    expect(out.elements_truncated).toBe(true);
    expect(out.collection.element_count).toBe(5);
  });

  it("stays quiet when the whole list fits", async () => {
    const out = await getCollectionDetails({ collectionId: "c1", maxElements: 9 }, ctxWith(clientFor(3)));
    expect(out.elements_truncated).toBe(false);
  });

  it("tolerates a collection whose members carry no object", async () => {
    const client = mockClient({
      GET: () => ({ data: { id: "c1", elements: [{}] }, response: { status: 200 } }),
    });
    const out = await getCollectionDetails({ collectionId: "c1" }, ctxWith(client));
    expect(out.elements[0]).toMatchObject({ element_index: 0, object_id: "", name: "" });
    expect(out.collection.state).toBe("unknown");
  });

  it("throws GalaxyNotFoundError on 404", async () => {
    const client = mockClient({ GET: () => ({ error: {}, response: { status: 404 } }) });
    await expect(getCollectionDetails({ collectionId: "x" }, ctxWith(client))).rejects.toBeInstanceOf(
      GalaxyNotFoundError,
    );
  });

  it("projects the shown count against the real one", () => {
    const msg = getCollectionDetailsOp.project!(
      { collection: { name: "reads", element_count: 5 }, elements: [{}, {}], collection_id: "c1" } as any,
      { collectionId: "c1" },
    );
    expect(msg.message).toBe("Collection reads (2 of 5 element(s))");
  });
});
