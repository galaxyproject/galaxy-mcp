import { z } from "zod";
import type { GetJson } from "../bindings";
import type { GalaxyContext } from "../context";
import { classifyHttp } from "../errors";
import { envelopeFact, readFact, recordFact } from "./envelope-facts";
import { register, runOperation } from "./registry";
import type { AnyOperation, InputOf, Operation } from "./types";

export type CollectionDetail = GetJson<"/api/dataset_collections/{hdca_id}">;

// Python's default, and like there it applies whether or not the caller asked.
const MAX_ELEMENTS = 100;

const input = {
  collectionId: z.string().describe("Encoded HDCA (history dataset collection) id"),
  maxElements: z
    .number()
    .int()
    .default(MAX_ELEMENTS)
    .describe(`Truncate the elements list to N (default ${MAX_ELEMENTS})`),
};
type In = { collectionId: string; maxElements?: number };

/**
 * How many elements the collection had before the list was cut.
 *
 * The envelope says whether it was cut, and the only way to know is to have seen
 * the untruncated list -- which run() no longer returns, because truncating is
 * what it is for. So the number travels beside the call, like list_pages' total.
 */
const rawElementCount = envelopeFact<number>("get_collection_details.raw_element_count");

/**
 * What `dict.get(key, fallback)` does: the fallback stands in for an ABSENT key and
 * for nothing else, so a field Galaxy sent as null stays null.
 */
function get(record: Record<string, unknown>, key: string, fallback: unknown): unknown {
  return key in record ? record[key] : fallback;
}

const asRecord = (value: unknown): Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};

/**
 * One element, flattened the way the other server flattens it.
 *
 * `index` is counted from the head of the list that is being RETURNED, and the
 * dataset behind the element is read exactly one level deep: a nested collection
 * is an element whose object is another collection, and it is left as one -- the
 * fields a dataset would fill come back empty rather than recursed into.
 */
function normalizeElement(element: unknown, index: number): Record<string, unknown> {
  const e = asRecord(element);
  const object = asRecord(get(e, "object", {}));
  return {
    element_index: index,
    element_identifier: get(e, "element_identifier", ""),
    element_type: get(e, "element_type", ""),
    object_id: get(object, "id", ""),
    name: get(object, "name", ""),
    state: get(object, "state", ""),
    extension: get(object, "extension", ""),
    file_size: get(object, "file_size", null),
  };
}

/** The sentence the other server puts on every answer, word for word. */
const ELEMENTS_NOTE =
  "Use get_dataset_details(object_id) to get full details " +
  "for individual datasets in this collection.";

async function run(i: In, ctx: GalaxyContext): Promise<CollectionDetail> {
  const { data, error, response } = await ctx.client.GET("/api/dataset_collections/{hdca_id}", {
    params: { path: { hdca_id: i.collectionId } },
  });
  if (error || !data) throw classifyHttp(response.status, error);
  // Always truncated, as on the Python side: a collection can hold thousands of
  // elements and a caller that did not ask for a limit is not asking for all of them.
  const d = data as { elements?: unknown[] };
  if (Array.isArray(d.elements)) {
    recordFact(ctx, rawElementCount, d.elements.length);
    d.elements = d.elements.slice(0, i.maxElements ?? MAX_ELEMENTS);
  } else {
    recordFact(ctx, rawElementCount, 0);
  }
  return data as CollectionDetail;
}

export const getCollectionDetailsOp: Operation<typeof input, CollectionDetail> = {
  name: "get_collection_details",
  domain: "collections",
  summary: "Show a dataset collection by id, with its elements (optionally truncated).",
  input,
  run,
  // The other server does not hand Galaxy's record back: it names the id that was
  // asked for, states the content type, keeps six fields of the collection itself
  // and flattens every element, then says whether the list was cut and which tool
  // opens one element. run() still returns the record, with the elements truncated,
  // which is what a library caller destructures.
  project: (c, i, facts) => {
    const record = asRecord(c);
    const elements = Array.isArray(record["elements"]) ? (record["elements"] as unknown[]) : [];
    const max = i.maxElements ?? MAX_ELEMENTS;
    const raw = readFact(facts, rawElementCount) ?? elements.length;
    return {
      data: {
        collection_id: i.collectionId,
        history_content_type: "dataset_collection",
        collection: {
          id: get(record, "id", null),
          name: get(record, "name", null),
          collection_type: get(record, "collection_type", null),
          element_count: get(record, "element_count", 0),
          populated: get(record, "populated", true),
          state: get(record, "state", "unknown"),
        },
        elements: elements.map(normalizeElement),
        elements_truncated: raw > max,
        note: ELEMENTS_NOTE,
      },
      message: `Collection ${String(get(record, "id", i.collectionId))} (${elements.length} elements)`,
      // The count is the elements actually returned -- after the truncation above --
      // which is what the other surface counts too.
      count: elements.length,
    };
  },
};

register(getCollectionDetailsOp as AnyOperation);

// A library caller may leave the defaulted arguments out; run() applies the same
// values the schema declares for the parsed surface path.
export const getCollectionDetails = (i: In, ctx: GalaxyContext) =>
  runOperation(getCollectionDetailsOp, i as InputOf<typeof input>, ctx);
