import { z } from "zod";
import type { GetJson } from "../bindings";
import type { GalaxyContext } from "../context";
import { classifyHttp } from "../errors";
import { register, runOperation } from "./registry";
import type { AnyOperation, InputOf, Operation } from "./types";

export type CollectionDetail = GetJson<"/api/dataset_collections/{hdca_id}">;

// Python's default, and like there it applies whether or not the caller asked.
const MAX_ELEMENTS = 100;

const input = {
  collectionId: z.string().describe("Encoded HDCA (history dataset collection) id"),
  maxElements: z.coerce
    .number()
    .int()
    .positive()
    .default(MAX_ELEMENTS)
    .describe(`Truncate the elements list to N (default ${MAX_ELEMENTS})`),
};
type In = { collectionId: string; maxElements?: number };

async function run(i: In, ctx: GalaxyContext): Promise<CollectionDetail> {
  const { data, error, response } = await ctx.client.GET("/api/dataset_collections/{hdca_id}", {
    params: { path: { hdca_id: i.collectionId } },
  });
  if (error || !data) throw classifyHttp(response.status, error);
  // Always truncated, as on the Python side: a collection can hold thousands of
  // elements and a caller that did not ask for a limit is not asking for all of them.
  const d = data as { elements?: unknown[] };
  if (Array.isArray(d.elements)) d.elements = d.elements.slice(0, i.maxElements ?? MAX_ELEMENTS);
  return data as CollectionDetail;
}

export const getCollectionDetailsOp: Operation<typeof input, CollectionDetail> = {
  name: "get_collection_details",
  domain: "collections",
  summary: "Show a dataset collection by id, with its elements (optionally truncated).",
  input,
  run,
  project: (c) => ({ message: `Collection ${(c as { id?: string }).id} (${((c as { elements?: unknown[] }).elements ?? []).length} elements)` }),
};

register(getCollectionDetailsOp as AnyOperation);

// A library caller may leave the defaulted arguments out; run() applies the same
// values the schema declares for the parsed surface path.
export const getCollectionDetails = (i: In, ctx: GalaxyContext) =>
  runOperation(getCollectionDetailsOp, i as InputOf<typeof input>, ctx);
