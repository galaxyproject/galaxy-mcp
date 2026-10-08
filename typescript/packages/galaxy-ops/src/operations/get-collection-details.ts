import { z } from "zod";
import type { GetJson } from "../bindings";
import type { GalaxyContext } from "../context";
import { httpError } from "../errors";
import { pyGet, pyStr } from "../python-values";
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
  const object = asRecord(pyGet(e, "object", {}));
  return {
    element_index: index,
    element_identifier: pyGet(e, "element_identifier", ""),
    element_type: pyGet(e, "element_type", ""),
    object_id: pyGet(object, "id", ""),
    name: pyGet(object, "name", ""),
    state: pyGet(object, "state", ""),
    extension: pyGet(object, "extension", ""),
    file_size: pyGet(object, "file_size", null),
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
  if (error || !data) throw httpError(response, error);
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
    const collection = {
      id: pyGet(record, "id", null),
      name: pyGet(record, "name", null),
      collection_type: pyGet(record, "collection_type", null),
      element_count: pyGet(record, "element_count", 0),
      populated: pyGet(record, "populated", true),
      state: pyGet(record, "state", "unknown"),
    };
    return {
      data: {
        collection_id: i.collectionId,
        history_content_type: "dataset_collection",
        collection,
        elements: elements.map(normalizeElement),
        elements_truncated: raw > max,
        note: ELEMENTS_NOTE,
      },
      // Read off the NORMALISED record, as the other server's sentence is
      // (server.py, get_collection_details): its `name` key is always there, so the
      // id it names as a fallback is unreachable and a collection Galaxy states no
      // name for is announced as `None`.
      message: `Retrieved collection '${pyStr(pyGet(collection, "name", i.collectionId))}'`,
      // The count is the elements actually returned -- after the truncation above --
      // which is what the other surface counts too.
      count: elements.length,
    };
  },
  // server.py, get_collection_details.
  failure: {
    shape: "bioblend-get",
    action: "Get collection details",
    context: (i) => ({ collection_id: i.collectionId }),
    sentence: (_text, status, i) =>
      status === 404
        ? `Collection ID '${i.collectionId}' not found. ` +
          "Make sure the collection exists and you have permission to view it."
        : undefined,
  },
};

register(getCollectionDetailsOp as AnyOperation);

// A library caller may leave the defaulted arguments out; run() applies the same
// values the schema declares for the parsed surface path.
export const getCollectionDetails = (i: In, ctx: GalaxyContext) =>
  runOperation(getCollectionDetailsOp, i as InputOf<typeof input>, ctx);
