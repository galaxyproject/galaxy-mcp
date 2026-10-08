import { describe, it, expect } from "vitest";
import { getCollectionDetailsOp, getCollectionDetails } from "../../src/operations/get-collection-details";
import { runWithEnvelope } from "../../src/operations/registry";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("get_collection_details", () => {
  it("shows a collection by hdca id and truncates elements", async () => {
    const client = mockClient({
      GET: (path, init) => {
        expect(path).toBe("/api/dataset_collections/{hdca_id}");
        expect(init.params.path.hdca_id).toBe("c1");
        return { data: { id: "c1", elements: [{ e: 1 }, { e: 2 }, { e: 3 }] }, response: { status: 200 } };
      },
    });
    const out = await getCollectionDetails({ collectionId: "c1", maxElements: 2 }, ctxWith(client));
    expect((out as any).elements.length).toBe(2);
  });

  const serving = (record: unknown) =>
    mockClient({ GET: () => ({ data: record, response: { status: 200 } }) });

  const NOTE =
    "Use get_dataset_details(object_id) to get full details " +
    "for individual datasets in this collection.";

  const envelope = (record: unknown, input: Record<string, unknown>) =>
    runWithEnvelope(getCollectionDetailsOp as never, input as never, ctxWith(serving(record)));

  /**
   * Every default at once. `dict.get(key, fallback)` substitutes for an ABSENT key
   * only, and the six fields of the collection do not share one fallback -- id, name
   * and collection_type go to null, element_count to 0, populated to true, state to
   * "unknown" -- so a reply that omits all of them is the only thing that says which
   * is which.
   */
  it("fills each missing field with its own default, not one shared blank", async () => {
    const r = await envelope({ elements: [{ element_identifier: "lonely", object: {} }] }, {
      collectionId: "c9",
    });
    expect(r.data).toEqual({
      collection_id: "c9",
      history_content_type: "dataset_collection",
      collection: {
        id: null,
        name: null,
        collection_type: null,
        element_count: 0,
        populated: true,
        state: "unknown",
      },
      elements: [
        {
          element_index: 0,
          element_identifier: "lonely",
          element_type: "",
          object_id: "",
          name: "",
          state: "",
          extension: "",
          file_size: null,
        },
      ],
      elements_truncated: false,
      note: NOTE,
    });
    expect(r.count).toBe(1);
  });

  /** A null Galaxy sent is a value, so the fallback must not eat it. */
  it("keeps a field Galaxy sent as null", async () => {
    const r = await envelope({ id: "c1", name: null, element_count: null, elements: [] }, {
      collectionId: "c1",
    });
    expect((r.data as any).collection).toMatchObject({ name: null, element_count: null });
  });

  it("says the list was cut, and counts only what it returned", async () => {
    const elements = Array.from({ length: 5 }, (_, k) => ({
      element_identifier: `s${k}`,
      element_type: "hda",
      object: { id: `d${k}`, name: `s${k}.txt`, state: "ok", extension: "txt", file_size: 7 },
    }));
    const cut = await envelope({ id: "c1", elements }, { collectionId: "c1", maxElements: 2 });
    expect((cut.data as any).elements_truncated).toBe(true);
    expect((cut.data as any).elements).toHaveLength(2);
    expect((cut.data as any).elements[1].element_index).toBe(1);
    expect(cut.count).toBe(2);

    const whole = await envelope({ id: "c1", elements }, { collectionId: "c1", maxElements: 5 });
    expect((whole.data as any).elements_truncated).toBe(false);
    expect(whole.count).toBe(5);
  });

  /** A nested collection is an element whose object is a collection, and stays one. */
  it("does not walk into a nested collection", async () => {
    const r = await envelope(
      {
        id: "c1",
        collection_type: "list:paired",
        elements: [
          {
            element_identifier: "pair0",
            element_type: "dataset_collection",
            object: { id: "c2", collection_type: "paired", elements: [{ element_identifier: "f" }] },
          },
        ],
      },
      { collectionId: "c1" },
    );
    expect((r.data as any).elements).toEqual([
      {
        element_index: 0,
        element_identifier: "pair0",
        element_type: "dataset_collection",
        object_id: "c2",
        name: "",
        state: "",
        extension: "",
        file_size: null,
      },
    ]);
  });

  /** The wrapper is the wire's; run() still answers with Galaxy's record. */
  it("leaves the library shape alone", async () => {
    const record = { id: "c1", name: "n", elements: [{ element_identifier: "a" }] };
    const out = await getCollectionDetails({ collectionId: "c1" }, ctxWith(serving(record)));
    expect(out).toEqual(record);
  });
});
