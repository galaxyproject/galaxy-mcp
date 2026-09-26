import { z } from "zod";
import type { GalaxyContext } from "../context";
import { classifyHttp } from "../errors";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

/** The collection's own metadata, without its elements. */
export interface CollectionSummary {
  id?: string;
  name?: string;
  collection_type?: string;
  element_count: number;
  populated: boolean;
  state: string;
}

/** One member, flattened: the element and the dataset it points at are one row. */
export interface CollectionElement {
  element_index: number;
  element_identifier: string;
  element_type: string;
  object_id: string;
  name: string;
  state: string;
  extension: string;
  file_size: unknown;
}

/** The Python tool's result shape. */
export interface CollectionDetail {
  collection_id: string;
  history_content_type: "dataset_collection";
  collection: CollectionSummary;
  elements: CollectionElement[];
  elements_truncated: boolean;
  note: string;
}

// Python's default, and like there it applies whether or not the caller asked.
const MAX_ELEMENTS = 100;
// The Python tool's sentence, so both surfaces answer alike.
const NOTE =
  "Use get_dataset_details(object_id) to get full details for individual datasets in this collection.";

const input = {
  collectionId: z.string().describe("Encoded HDCA (history dataset collection) id"),
  maxElements: z
    .number()
    .int()
    .default(MAX_ELEMENTS)
    .describe(`Truncate the elements list to N (default ${MAX_ELEMENTS})`),
};
type In = { collectionId: string; maxElements?: number };

/** Galaxy's element: the membership, with the dataset nested inside it. */
interface RawElement {
  element_identifier?: string;
  element_type?: string;
  object?: { id?: string; name?: string; state?: string; extension?: string; file_size?: unknown };
}

async function run(i: In, ctx: GalaxyContext): Promise<CollectionDetail> {
  const { data, error, response } = await ctx.client.GET("/api/dataset_collections/{hdca_id}", {
    params: { path: { hdca_id: i.collectionId } },
  });
  if (error || !data) throw classifyHttp(response.status, error);
  const raw = data as {
    id?: string;
    name?: string;
    collection_type?: string;
    element_count?: number;
    populated?: boolean;
    state?: string;
    elements?: RawElement[];
  };

  // Report truncation so callers can distinguish a bounded result from a small collection.
  const max = i.maxElements ?? MAX_ELEMENTS;
  const all = Array.isArray(raw.elements) ? raw.elements : [];
  const elements = all.slice(0, max).map((element, element_index) => {
    const object = element.object ?? {};
    return {
      element_index,
      element_identifier: element.element_identifier ?? "",
      element_type: element.element_type ?? "",
      object_id: object.id ?? "",
      name: object.name ?? "",
      state: object.state ?? "",
      extension: object.extension ?? "",
      file_size: object.file_size,
    };
  });

  return {
    collection_id: i.collectionId,
    history_content_type: "dataset_collection",
    collection: {
      id: raw.id,
      name: raw.name,
      collection_type: raw.collection_type,
      element_count: raw.element_count ?? 0,
      populated: raw.populated ?? true,
      state: raw.state ?? "unknown",
    },
    elements,
    elements_truncated: all.length > max,
    note: NOTE,
  };
}

export const getCollectionDetailsOp: Operation<typeof input, CollectionDetail> = {
  name: "get_collection_details",
  domain: "collections",
  summary: "Show a dataset collection by id: its metadata and its members, flattened and truncated.",
  input,
  run,
  project: (c) => ({
    message: `Collection ${c.collection.name ?? c.collection_id} (${c.elements.length} of ${c.collection.element_count} element(s))`,
  }),
};

register(getCollectionDetailsOp as AnyOperation);

export const getCollectionDetails = (i: In, ctx: GalaxyContext) => runOperation(getCollectionDetailsOp, i, ctx);
